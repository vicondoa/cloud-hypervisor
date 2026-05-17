// Copyright 2019 Intel Corporation. All Rights Reserved.
// Copyright 2022 Unikie
// Copyright 2023 Alyssa Ross <hi@alyssa.is>
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::sync::{Arc, Barrier, Mutex};
use std::{result, thread};

use event_monitor::event;
use log::error;
use seccompiler::SeccompAction;
use vhost::vhost_user::message::{
    VhostSharedMemoryRegion, VhostUserConfigFlags, VhostUserProtocolFeatures, VhostUserMMap,
    VhostUserVirtioFeatures,
};
use vhost::vhost_user::{
    FrontendReqHandler, HandlerResult, VhostUserFrontend, VhostUserFrontendReqHandler,
};
use virtio_bindings::virtio_config::VIRTIO_F_ACCESS_PLATFORM;
use virtio_bindings::virtio_gpu::{
    VIRTIO_GPU_F_CONTEXT_INIT, VIRTIO_GPU_F_RESOURCE_BLOB, VIRTIO_GPU_F_RESOURCE_UUID,
    VIRTIO_GPU_F_VIRGL,
};
use virtio_queue::Queue;
use vm_device::UserspaceMapping;
use vm_memory::volatile_memory::PtrGuardMut;
use vm_memory::{GuestMemoryAtomic, VolatileMemory};
use vm_migration::{MigratableError, Pausable};
use vmm_sys_util::eventfd::EventFd;

use super::vu_common_ctrl::VhostUserHandle;
use super::{Error, Result};
use crate::seccomp_filters::Thread;
use crate::thread_helper::spawn_virtio_thread;
use crate::vhost_user::VhostUserCommon;
use crate::{
    ActivateError, ActivateResult, GuestMemoryMmap, GuestRegionMmap, MmapRegion,
     VIRTIO_F_VERSION_1, VirtioCommon, VirtioDevice, VirtioDeviceType,
    VirtioInterrupt, VirtioSharedMemoryList,
};

const QUEUE_SIZES: &[u16] = &[256, 16];
const NUM_QUEUES: u16 = QUEUE_SIZES.len() as _;

struct BackendReqHandler {
    mapping: Arc<MmapRegion>,
}

impl BackendReqHandler {
    fn ptr_guard_mut(&self, offset: u64, len: u64) -> io::Result<PtrGuardMut> {
        let shm_offset = offset
            .try_into()
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        let len = len
            .try_into()
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?;
        Ok(self
            .mapping
            .get_slice(shm_offset, len)
            .map_err(|_| io::Error::from_raw_os_error(libc::EINVAL))?
            .ptr_guard_mut())
    }
}

impl VhostUserFrontendReqHandler for BackendReqHandler {
    fn shmem_map(&self, req: &VhostUserMMap, fd: &dyn AsRawFd) -> HandlerResult<u64> {
        let target = self.ptr_guard_mut(req.shm_offset, req.len)?;

        // SAFETY: we've checked we're only giving addr and length
        // within the region, and are passing MAP_FIXED to ensure they
        // are respected.
        // crosvm sends VhostUserMMapFlags::WRITABLE = 1<<0 to mean
        // "writable mapping"; v0.22 vhost defines no other bits. Translate
        // to libc::PROT_*. The spectrum-50 code treated `req.flags as i32`
        // as raw PROT_* bits, but that interpretation matches NEITHER the
        // crosvm bit layout NOR the v0.22 layout and produces non-writable
        // mappings that the guest EFAULTs on first write.
        let writable = (req.flags & 1) != 0;  // VhostUserMMapFlags::WRITABLE
        let prot: i32 = libc::PROT_READ | (if writable { libc::PROT_WRITE } else { 0 });
        let ret = unsafe {
            libc::mmap(
                target.as_ptr().cast(),
                target.len(),
                prot,
                // https://bugzilla.kernel.org/show_bug.cgi?id=217238
                if prot & libc::PROT_WRITE != 0 {
                    libc::MAP_SHARED
                } else {
                    libc::MAP_PRIVATE
                } | libc::MAP_FIXED,
                fd.as_raw_fd(),
                req.fd_offset as libc::off_t,
            )
        };

        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(0)
    }

    fn shmem_unmap(&self, req: &VhostUserMMap) -> HandlerResult<u64> {
        let target = self.ptr_guard_mut(req.shm_offset, req.len)?;

        // SAFETY: we've checked we're only giving addr and length
        // within the region, and are passing MAP_FIXED to ensure they
        // are respected.
        let ret = unsafe {
            libc::mmap(
                target.as_ptr().cast(),
                target.len(),
                libc::PROT_NONE,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if ret == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }

        Ok(0)
    }
}

pub struct Gpu {
    common: VirtioCommon,
    vu_common: VhostUserCommon,
    id: String,
    // Hold ownership of the memory that is allocated for the device
    // which will be automatically dropped when the device is dropped
    cache: Option<VirtioSharedMemoryList>,
    backend_req_support: bool,
    seccomp_action: SeccompAction,
    guest_memory: Option<GuestMemoryAtomic<GuestMemoryMmap>>,
    epoll_thread: Option<thread::JoinHandle<()>>,
    exit_evt: EventFd,
    iommu: bool,
}

impl Gpu {
    /// Create a new virtio-gpu device.
    pub fn new(
        id: String,
        path: &str,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        iommu: bool,
    ) -> Result<(Gpu, VhostSharedMemoryRegion)> {
        // Connect to the vhost-user socket.
        let mut vu = VhostUserHandle::connect_vhost_user(false, path, NUM_QUEUES as u64, false, None)?;

        let avail_features = 1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_GPU_F_VIRGL
            | 1 << VIRTIO_GPU_F_RESOURCE_UUID
            | 1 << VIRTIO_GPU_F_RESOURCE_BLOB
            | 1 << VIRTIO_GPU_F_CONTEXT_INIT
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits();

        let avail_protocol_features = VhostUserProtocolFeatures::CONFIG
            | VhostUserProtocolFeatures::BACKEND_REQ
            | VhostUserProtocolFeatures::SHMEM_MAP_CROSVM
            | VhostUserProtocolFeatures::REPLY_ACK;

        let (acked_features, acked_protocol_features) =
            vu.negotiate_features_vhost_user(avail_features, avail_protocol_features)?;

        let shm_regions = vu.get_shared_memory_regions()?;
        if shm_regions.len() != 1 {
            return Err(Error::VhostUserUnexpectedSharedMemoryRegionsCount(
                1,
                shm_regions.len(),
            ));
        }
        let shm_region = shm_regions[0];

        Ok((
            Gpu {
                common: VirtioCommon {
                    device_type: VirtioDeviceType::Gpu as u32,
                    avail_features: acked_features,
                    // If part of the available features that have been acked, the
                    // PROTOCOL_FEATURES bit must be already set through the VIRTIO
                    // acked features as we know the guest would never ack it, this
                    // the feature would be lost.
                    acked_features: acked_features
                        & VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits(),
                    paused_sync: Some(Arc::new(Barrier::new(NUM_QUEUES as usize))),
                    queue_sizes: QUEUE_SIZES.to_vec(),
                    min_queues: NUM_QUEUES,
                    ..Default::default()
                },
                vu_common: VhostUserCommon {
                    vu: Some(Arc::new(Mutex::new(vu))),
                    acked_protocol_features,
                    socket_path: path.to_string(),
                    vu_num_queues: NUM_QUEUES as usize,
                    ..Default::default()
                },
                id,
                cache: None,
                backend_req_support: acked_protocol_features
                    & VhostUserProtocolFeatures::BACKEND_REQ.bits()
                    != 0,
                seccomp_action,
                guest_memory: None,
                epoll_thread: None,
                exit_evt,
                iommu,
            },
            shm_region,
        ))
    }

    pub fn set_cache(&mut self, cache: VirtioSharedMemoryList) {
        self.cache = Some(cache);
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        if let Some(kill_evt) = self.common.kill_evt.take() {
            // Ignore the result because there is nothing we can do about it.
            let _ = kill_evt.write(1);
        }
    }
}

impl VirtioDevice for Gpu {
    fn device_type(&self) -> u32 {
        self.common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.common.queue_sizes
    }

    fn features(&self) -> u64 {
        let mut features = self.common.avail_features;
        if self.iommu {
            features |= 1u64 << VIRTIO_F_ACCESS_PLATFORM;
        }
        features
    }

    fn ack_features(&mut self, value: u64) {
        self.common.ack_features(value);
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        if let Some(vu) = &self.vu_common.vu
            && let Err(e) = vu
                .lock()
                .unwrap()
                .socket_handle()
                .get_config(
                    offset as u32,
                    data.len() as u32,
                    VhostUserConfigFlags::WRITABLE,
                    data,
                )
                .map_err(|e| format!("{e:?}"))
                .and_then(|(_, config)| data.write_all(&config).map_err(|e| format!("{e:?}")))
        {
            error!("Failed getting vhost-user-gpu configuration: {e:?}");
        }
    }

    fn activate(&mut self, context: crate::device::ActivationContext) -> ActivateResult {
        let crate::device::ActivationContext {
            mem,
            interrupt_cb,
            queues,
            device_status,
        } = context;
        self.common.activate(&queues, interrupt_cb.clone())?;
        self.guest_memory = Some(mem.clone());

        // Initialize backend communication.
        let backend_req_handler = if self.backend_req_support {
            if let Some(cache) = self.cache.as_ref() {
                let vu_frontend_req_handler = Arc::new(BackendReqHandler {
                    mapping: cache.mapping.clone(),
                });

                let mut req_handler =
                    FrontendReqHandler::new(vu_frontend_req_handler).map_err(|e| {
                        ActivateError::VhostUserGpuSetup(Error::FrontendReqHandlerCreation(e))
                    })?;

                if self.vu_common.acked_protocol_features
                    & VhostUserProtocolFeatures::REPLY_ACK.bits()
                    != 0
                {
                    req_handler.set_reply_ack_flag(true);
                }

                Some(req_handler)
            } else {
                None
            }
        } else {
            None
        };

        // Run a dedicated thread for handling potential reconnections with
        // the backend.
        let (kill_evt, pause_evt) = self.common.dup_eventfds();

        let mut handler = self.vu_common.activate(
            mem,
            &queues,
            interrupt_cb.clone(),
            self.common.acked_features,
            backend_req_handler,
            kill_evt,
            pause_evt,
        )?;

        let paused = self.common.paused.clone();
        let paused_sync = self.common.paused_sync.clone();

        let mut epoll_threads = Vec::new();
        spawn_virtio_thread(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioVhostGpu,
            &mut epoll_threads,
            &self.exit_evt,
            device_status.clone(),
            interrupt_cb.clone(),
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        )?;
        self.epoll_thread = Some(epoll_threads.remove(0));

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) {
        // We first must resume the virtio thread if it was paused.
        if self.common.pause_evt.take().is_some() {
            let _ = self.common.resume();
        }

        if let Some(vu) = &self.vu_common.vu
            && let Err(e) = vu.lock().unwrap().reset_vhost_user()
        {
            error!("Failed to reset vhost-user daemon: {e:?}");
            return;
        }

        if let Some(kill_evt) = self.common.kill_evt.take() {
            // Ignore the result because there is nothing we can do about it.
            let _ = kill_evt.write(1);
        }

        event!("virtio-device", "reset", "id", &self.id);

        // Return the interrupt
        // dropped: caller no longer gets an interrupt_cb back
    }

    fn shutdown(&mut self) {
        self.vu_common.shutdown();
    }

    fn get_shm_regions(&self) -> Option<VirtioSharedMemoryList> {
        self.cache.clone()
    }

    fn set_shm_regions(
        &mut self,
        shm_regions: VirtioSharedMemoryList,
    ) -> std::result::Result<(), crate::Error> {
        if let Some(cache) = self.cache.as_mut() {
            *cache = shm_regions;
            Ok(())
        } else {
            Err(crate::Error::SetShmRegionsNotSupported)
        }
    }

    fn add_memory_region(
        &mut self,
        region: &Arc<GuestRegionMmap>,
    ) -> std::result::Result<(), crate::Error> {
        self.vu_common.add_memory_region(&self.guest_memory, region)
    }

    fn userspace_mappings(&self) -> Vec<UserspaceMapping> {
        let mut mappings = Vec::new();
        if let Some(cache) = self.cache.as_ref() {
            mappings.push(UserspaceMapping {
                mapping: cache.mapping.clone(),
                mem_slot: cache.mem_slot,
                addr: cache.addr,
                mergeable: false,
            });
        }

        mappings
    }
}

impl Pausable for Gpu {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.pause()?;
        self.common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        self.common.resume()?;

        if let Some(epoll_thread) = &self.epoll_thread {
            epoll_thread.thread().unpark();
        }

        self.vu_common.resume()
    }
}
