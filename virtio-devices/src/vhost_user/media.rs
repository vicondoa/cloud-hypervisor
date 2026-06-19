// Copyright © 2025 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
use std::io;
use std::os::unix::io::AsRawFd;
use std::result;
use std::sync::{Arc, Barrier, Mutex};

use event_monitor::event;
use seccompiler::SeccompAction;
use vhost::vhost_user::message::{
    VhostSharedMemoryRegion, VhostUserConfigFlags, VhostUserMMap, VhostUserProtocolFeatures,
    VhostUserVirtioFeatures,
};
use vhost::vhost_user::{
    FrontendReqHandler, HandlerResult, VhostUserFrontend, VhostUserFrontendReqHandler,
};
use vm_device::UserspaceMapping;
use vm_memory::volatile_memory::PtrGuardMut;
use vm_memory::{GuestMemoryAtomic, VolatileMemory};
use vm_migration::{MigratableError, Pausable};
use vmm_sys_util::eventfd::EventFd;

use super::vu_common_ctrl::VhostUserHandle;
use super::{DEFAULT_VIRTIO_FEATURES, Error, Result};
use crate::seccomp_filters::Thread;
use crate::thread_helper::spawn_virtio_thread;
use crate::vhost_user::VhostUserCommon;
use crate::{
    ActivateError, ActivateResult, GuestMemoryMmap, GuestRegionMmap, MmapRegion,
    VIRTIO_F_ACCESS_PLATFORM, VirtioDevice, VirtioSharedMemoryList,
};

const VIRTIO_ID_MEDIA: u32 = 48;
const QUEUE_SIZES: &[u16] = &[256, 256];
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
        let writable = (req.flags & 1) != 0; // VhostUserMMapFlags::WRITABLE
        let prot: i32 = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let ret = unsafe {
            libc::mmap(
                target.as_ptr().cast(),
                target.len(),
                prot,
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

pub struct Media {
    vu_common: VhostUserCommon,
    id: String,
    cache: Option<VirtioSharedMemoryList>,
    seccomp_action: SeccompAction,
    guest_memory: Option<GuestMemoryAtomic<GuestMemoryMmap>>,
    exit_evt: EventFd,
    access_platform_enabled: bool,
}

impl Media {
    pub fn new(
        id: String,
        path: &str,
        seccomp_action: SeccompAction,
        exit_evt: EventFd,
        access_platform_enabled: bool,
    ) -> Result<(Self, VhostSharedMemoryRegion)> {
        let mut vu =
            VhostUserHandle::connect_vhost_user(false, path, NUM_QUEUES as u64, false, None)?;

        let avail_protocol_features = VhostUserProtocolFeatures::CONFIG
            | VhostUserProtocolFeatures::BACKEND_REQ
            | VhostUserProtocolFeatures::SHMEM_MAP_CROSVM
            | VhostUserProtocolFeatures::REPLY_ACK;

        let (acked_features, acked_protocol_features) =
            vu.negotiate_features_vhost_user(DEFAULT_VIRTIO_FEATURES, avail_protocol_features)?;

        let shm_region = VhostSharedMemoryRegion { id: 0, padding: [0; 7], length: 256 * 1024 * 1024 };

        Ok((
            Self {
                vu_common: VhostUserCommon {
                    virtio_common: crate::VirtioCommon {
                        device_type: VIRTIO_ID_MEDIA,
                        avail_features: acked_features,
                        acked_features: acked_features
                            & VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits(),
                        paused_sync: Some(Arc::new(Barrier::new(NUM_QUEUES as usize))),
                        queue_sizes: QUEUE_SIZES.to_vec(),
                        min_queues: NUM_QUEUES,
                        ..Default::default()
                    },
                    vu: Some(Arc::new(Mutex::new(vu))),
                    acked_protocol_features,
                    socket_path: path.to_string(),
                    vu_num_queues: NUM_QUEUES as usize,
                    ..Default::default()
                },
                id,
                cache: None,
                seccomp_action,
                guest_memory: None,
                exit_evt,
                access_platform_enabled,
            },
            shm_region,
        ))
    }

    pub fn set_cache(&mut self, cache: VirtioSharedMemoryList) {
        self.cache = Some(cache);
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        self.vu_common.shutdown();
    }
}

impl VirtioDevice for Media {
    fn device_type(&self) -> u32 {
        self.vu_common.virtio_common.device_type
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &self.vu_common.virtio_common.queue_sizes
    }

    fn features(&self) -> u64 {
        let mut features = self.vu_common.virtio_common.avail_features;
        if self.access_platform_enabled {
            features |= 1u64 << VIRTIO_F_ACCESS_PLATFORM;
        }
        features
    }

    fn ack_features(&mut self, value: u64) {
        self.vu_common.virtio_common.ack_features(value);
    }

    fn read_config(&self, offset: u64, mut data: &mut [u8]) {
        if let Some(vu) = &self.vu_common.vu {
            if let Err(e) = vu
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
                .and_then(|(_, config)| {
                    use std::io::Write;
                    data.write_all(&config).map_err(|e| format!("{e:?}"))
                })
            {
                log::error!("Failed getting vhost-user-media configuration: {e:?}");
            }
        }
    }

    fn activate(&mut self, context: crate::device::ActivationContext) -> ActivateResult {
        let crate::device::ActivationContext {
            mem,
            interrupt_cb,
            queues,
            device_status,
        } = context;
        self.vu_common
            .virtio_common
            .activate(&queues, interrupt_cb.clone())?;
        self.guest_memory = Some(mem.clone());

        let backend_req_handler = if self.vu_common.acked_protocol_features
            & VhostUserProtocolFeatures::BACKEND_REQ.bits()
            != 0
        {
            if let Some(cache) = self.cache.as_ref() {
                let vu_frontend_req_handler = Arc::new(BackendReqHandler {
                    mapping: cache.mapping.clone(),
                });

                let mut req_handler =
                    FrontendReqHandler::new(vu_frontend_req_handler).map_err(|e| {
                        ActivateError::VhostUserSetup(Error::FrontendReqHandlerCreation(e))
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

        let (kill_evt, pause_evt) = self.vu_common.virtio_common.dup_eventfds();

        // Force SET_VRING_BASE(0) for all queues. The virtio-media guest driver
        // pre-queues event buffers on queue 1 before DRIVER_OK, so by the time
        // activate() runs, avail_idx is already non-zero. Without this override,
        // setup_vhost_user reads avail_idx from guest memory and uses it as the
        // base, making the pre-queued event buffers invisible to the backend.
        if self.vu_common.vring_bases.is_none() {
            self.vu_common.vring_bases = Some(vec![0; queues.len()]);
        }

        let mut handler = self.vu_common.activate(
            mem,
            &queues,
            interrupt_cb.clone(),
            self.vu_common.virtio_common.acked_features,
            backend_req_handler,
            kill_evt,
            pause_evt,
        )?;

        let paused = self.vu_common.virtio_common.paused.clone();
        let paused_sync = self.vu_common.virtio_common.paused_sync.clone();

        let mut epoll_threads = Vec::new();
        spawn_virtio_thread(
            &self.id,
            &self.seccomp_action,
            Thread::VirtioVhostMedia,
            &mut epoll_threads,
            &self.exit_evt,
            device_status.clone(),
            interrupt_cb.clone(),
            move || handler.run(&paused, paused_sync.as_ref().unwrap()),
        )?;
        self.vu_common.epoll_thread = Some(epoll_threads.remove(0));

        event!("virtio-device", "activated", "id", &self.id);
        Ok(())
    }

    fn reset(&mut self) {
        self.vu_common.reset(&self.id);
    }

    fn shutdown(&mut self) {
        self.vu_common.shutdown();
    }

    fn get_shm_regions(&self) -> Option<VirtioSharedMemoryList> {
        let has = self.cache.is_some();
        log::warn!("media get_shm_regions called: cache={has}");
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
        // SHM is registered with KVM via create_userspace_mapping in make_media_device
        // and cleaned up by the OS on VM shutdown. Not returned here because vhost-user
        // devices send all userspace_mappings via SET_MEM_TABLE during activate(), and
        // the SHM anonymous mapping (PROT_NONE, no file backing) breaks the backend's
        // GuestMemory resolution.
        Vec::new()
    }
}

impl Pausable for Media {
    fn pause(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.pause()?;
        self.vu_common.virtio_common.pause()
    }

    fn resume(&mut self) -> result::Result<(), MigratableError> {
        self.vu_common.virtio_common.resume()?;

        if let Some(epoll_thread) = &self.vu_common.epoll_thread {
            epoll_thread.thread().unpark();
        }

        self.vu_common.resume()
    }
}
