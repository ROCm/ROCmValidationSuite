/*******************************************************************************
 * Copyright (c) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
 ********************************************************************************/
#include <iostream>
#include <atomic>

#include <sys/types.h>
#include <sys/mman.h>

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <string.h>
#include <unistd.h>

#include <pciaccess.h>

#define MAJOR_VERSION 1
#define MINOR_VERSION 0

static void print_help()
{
    std::cerr << "AMD EDP Helper Tool version ";
    std::cerr << MAJOR_VERSION << "." << MINOR_VERSION << std::endl;
    std::cerr << std::endl;
    std::cerr << "This tool requires root access, as it uses /dev/mem to access the GPU." << std::endl;
    std::cerr << "It will also interact poorly with the ROCm debugging tools such as GDB." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Command line parameters:" << std::endl;
    std::cerr << "   -h, --help: Print this help menu." << std::endl;
    std::cerr << "   -s #, --sleep #: Number of microseconds to delay between stopping all waves and trying to restart them. (default 1000)" << std::endl;
    std::cerr << "   -l #, --loops #: Number of times to loop through the process of stopping/restarting waves on the GPUs. (default 10000)" << std::endl;
    std::cerr << "   -d #, --delay #: Number of microseconds to delay between starting all waves and exiting the application. (default 1000)" << std::endl;
}

static void check_opts(const int argc, char** argv, uint32_t * sleep_us,
        uint32_t * loops, uint32_t * loop_delay_us, bool * debug)
{
    const char* const opts = "d:l:s:hD";
    const struct option long_opts[] = {
            {"help", 0, NULL, 'h'},
            {"delay", 1, NULL, 'd'},
            {"loops", 1, NULL, 'l'},
            {"sleep", 1, NULL, 's'},
            {"debug", 0, NULL, 'D'},
            {NULL, 0, NULL, 0}
    };

    if (argv == NULL || sleep_us == NULL || loops == NULL ||
            loop_delay_us == NULL || debug == NULL)
    {
        std::cerr << "Incorrectly passing arguments." << std::endl;
        std::cerr << "Pointers were: " <<
                (void*)argv << " " << (void*)sleep_us << " " <<
                (void*)loops << " " << (void*)loop_delay_us << std::endl;
        exit(-1);
    }

    *sleep_us = 1000;
    *loops = 10000;
    *loop_delay_us = 1000;
    *debug = false;

    while (1)
    {
        int retval = getopt_long(argc, argv, opts, long_opts, NULL);
        if (retval == -1)
            break;;
        switch (retval)
        {
            case 'd':
                *loop_delay_us = (uint32_t)strtol(optarg, NULL, 0);
                break;
            case 'l':
                *loops = (uint32_t)strtol(optarg, NULL, 0);
                break;
            case 's':
                *sleep_us = (uint32_t)strtol(optarg, NULL, 0);
                break;
            case 'D':
                *debug = true;
                break;
            case 'h':
            case '?':
            default:
                print_help();
                exit(-1);
        }
    }
}

static void print_gpu_name(int bus, int dev, int func,
        struct pci_device *pci_dev)
{
    const char *dev_name;
    dev_name = pci_device_get_device_name(pci_dev);
    std::cout << std::hex;
    std::cout << "AMD GPU found at Bus: " << (uint32_t)bus << " ";
    std::cout << "Dev: " << (uint32_t)dev << " ";
    std::cout << "Func: " << (uint32_t)func << std::endl;
    if (dev_name == NULL)
    {
        std::cout << "    Unlisted device with device ID ";
        std::cout << pci_dev->device_id << std::endl;
    }
    else
        std::cout << "    " << dev_name << std::endl;
}

static bool check_for_supported_amd_dev(struct pci_device *dev)
{
    if (dev->vendor_id != 0x1002)
        return false;

    if (dev->device_id >= 0x67c0 && dev->device_id <= 0x67df)
    {
        // Polaris 10 / Ellesmere
        return true;
    }
    else if (dev->device_id >= 0x67e0 && dev->device_id <= 0x67ff)
    {
        // Polaris 11 / Baffin / RacerX
        return true;
    }
    else if (dev->device_id >= 0x6940 && dev->device_id <= 0x695f)
    {
        // Polaris 22 / Vega M
        return true;
    }
    else if (dev->device_id >= 0x6980 && dev->device_id <= 0x699f)
    {
        // Polaris 12 / Lexa
        return true;
    }
    else if (dev->device_id >= 0x69a0 && dev->device_id <= 0x69bf)
    {
        // Vega 12
        return true;
    }
    else if (dev->device_id >= 0x7300 && dev->device_id <= 0x733f)
    {
        // Fiji
        return true;
    }
    else if (dev->device_id >= 0x6860 && dev->device_id <= 0x687f)
    {
        // Vega 10 / Greenland
        return true;
    }
    else if (dev->device_id == 0x15dd)
    {
        // Raven
        return true;
    }
    else if (dev->device_id >= 0x66a0 && dev->device_id <= 0x66bf)
    {
        // Vega 20
        return true;
    }
    else if (dev->device_id >= 0x7380 && dev->device_id <= 0x739f)
    {
        // MI100
        return true;
    }
    return false;
}

static uint32_t count_amd_gpus(void)
{
    int err = pci_system_init();
    if (err != 0)
    {
        fprintf(stderr, "Problem opening PCI System -- %s\n",
                strerror(err));
        exit(-1);
    }

    struct pci_device *dev;
    uint32_t num_amd_devices = 0;

    struct pci_device_iterator * iter = pci_slot_match_iterator_create(NULL);

    while ((dev = pci_device_next(iter)) != NULL)
    {
        if (check_for_supported_amd_dev(dev))
            num_amd_devices++;
    }

    pci_system_cleanup();

    return num_amd_devices;
}

typedef struct gpu_device {
    int32_t bus;
    int32_t device;
    int32_t function;

    pciaddr_t pci_base_addr;
    pciaddr_t pci_base_size;

    void* devmem_addr;

    uint32_t sq_cmd_offset;
    uint32_t *sq_cmd_addr;
    uint32_t grbm_gfx_index_offset;
    uint32_t *grbm_gfx_index_addr;
} gpu_device_t;

typedef struct func_arg {
    int which_gpu;
    uint64_t start_ns;
    uint64_t restart_ns;

    uint32_t sleep_us;
} func_arg_t;

uint32_t num_gpus;
gpu_device_t *gpus;

pthread_t *child_threads;

std::atomic_uint spinval1, spinval2;

static void find_amd_gpus(void)
{
    int err = pci_system_init();
    if (err != 0)
    {
        fprintf(stderr, "Problem opening PCI System -- %s\n",
                strerror(err));
        exit(-1);
    }

    struct pci_device *dev;
    int num = 0;

    struct pci_device_iterator * iter = pci_slot_match_iterator_create(NULL);

    while ((dev = pci_device_next(iter)) != NULL)
    {
        if (check_for_supported_amd_dev(dev))
        {
            print_gpu_name(dev->bus, dev->dev, dev->func, dev);

            gpus[num].bus = dev->bus;
            gpus[num].device = dev->dev;
            gpus[num].function = dev->func;

            // Config registers we care about are in the 6th BAR region
            pci_device_probe(dev);
            gpus[num].pci_base_addr = dev->regions[5].base_addr;
            gpus[num].pci_base_size = dev->regions[5].size;

            gpus[num].sq_cmd_offset = 0x8dec;
            gpus[num].grbm_gfx_index_offset = 0x30800;
            num++;
        }
    }
    pci_system_cleanup();
}

static void halt_sq(int gpu_num)
{
    *(gpus[gpu_num].grbm_gfx_index_addr) = 0xe0000000;
    *(gpus[gpu_num].sq_cmd_addr) = 0x111;
}

static void restart_sq(int gpu_num)
{
    *(gpus[gpu_num].grbm_gfx_index_addr) = 0xe0000000;
    for (int slot = 0; slot < 10; slot++)
    {
        for (int simd = 0; simd < 4; simd++)
        {
            const uint32_t halt_cmd = 0x1 | (slot << 16) | (simd << 20);
            *(gpus[gpu_num].sq_cmd_addr) = halt_cmd;
        }
    }
}

void mask_sig(void)
{
    sigset_t mask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);
    sigaddset(&mask, SIGQUIT);
    sigaddset(&mask, SIGHUP);

    pthread_sigmask(SIG_BLOCK, &mask, NULL);

}

static void sig_handler(int signo)
{
    if (signo == SIGINT || signo == SIGTERM || signo == SIGQUIT ||
            signo == SIGHUP)
    {
        spinval1.store(num_gpus, std::memory_order_seq_cst);
        spinval2.store(num_gpus, std::memory_order_seq_cst);

        for (uint32_t i = 0; i < num_gpus; i++)
        {
            pthread_join(child_threads[i], NULL);
            restart_sq(i);
        }
        std::cout << std::endl;
        exit(0);
    }
}

static void * pause_and_restart_waves(void *arg)
{
    struct timespec start_time, restart_time;
    func_arg_t *f_args = (func_arg_t*)arg;
    int which_gpu = f_args->which_gpu;
    uint64_t *start_ns = &(f_args->start_ns);
    uint64_t *restart_ns = &(f_args->restart_ns);

    mask_sig();

    halt_sq(which_gpu);

    spinval1.fetch_add(1, std::memory_order_release);
    while (spinval1.load(std::memory_order_relaxed) < num_gpus);
    std::atomic_thread_fence(std::memory_order_acquire);

    if (f_args->sleep_us != 0)
    {
        struct timespec start_delay_time, cur_delay_time;
        uint64_t start_delay_ns, cur_delay_ns;
        clock_gettime(CLOCK_MONOTONIC, &start_delay_time);
        start_delay_ns = 1000000000ULL * start_delay_time.tv_sec + start_delay_time.tv_nsec;
        do {
            halt_sq(which_gpu);
            clock_gettime(CLOCK_MONOTONIC, &cur_delay_time);
            cur_delay_ns = 1000000000ULL * cur_delay_time.tv_sec + cur_delay_time.tv_nsec;
        } while (cur_delay_ns - start_delay_ns < (f_args->sleep_us * 1000ULL));
    }

    spinval2.fetch_add(1, std::memory_order_release);
    while (spinval2.load(std::memory_order_relaxed) < num_gpus);
    std::atomic_thread_fence(std::memory_order_acquire);

    clock_gettime(CLOCK_MONOTONIC, &start_time);
    restart_sq(which_gpu);
    clock_gettime(CLOCK_MONOTONIC, &restart_time);

    *start_ns = 1000000000ULL * start_time.tv_sec + start_time.tv_nsec;
    *restart_ns = 1000000000ULL * restart_time.tv_sec + restart_time.tv_nsec;

    return NULL;
}

int main(int argc, char** argv)
{
    uint32_t sleep_us;
    uint32_t loops;
    uint32_t loop_delay_us;
    bool debug;

    /*************************************************************************/
    /* Parse the command line parameters *************************************/
    // Arguments to pull out of the command line.
    check_opts(argc, argv, &sleep_us, &loops, &loop_delay_us, &debug);

    num_gpus = count_amd_gpus();
    if (num_gpus == 0)
    {
        std::cerr << "ERROR. No supported AMD GPUs found." << std::endl;
        exit(-1);
    }

    gpus = (gpu_device_t*)calloc(num_gpus, sizeof(gpu_device_t));

    find_amd_gpus();

    child_threads = (pthread_t*)malloc(num_gpus * sizeof(pthread_t));

    // Open up /dev/mem so we can directly access the MMIO physical memory
    // region from within this application.
    // Requires root access!
    int devmem_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (devmem_fd == -1) {
        fprintf(stderr, "Error opening /dev/mem (%d) : %s\n", errno,
                strerror(errno));
        free(gpus);
        return errno;
    }
    int mmap_flags = PROT_READ | PROT_WRITE;
    for (uint32_t i = 0; i < num_gpus; i++)
    {
        gpus[i].devmem_addr = mmap(0, gpus[i].pci_base_size, mmap_flags, MAP_SHARED,
                devmem_fd, gpus[i].pci_base_addr);

        // Figure out MMIO adresses for all of the config registers we care about.
        gpus[i].sq_cmd_addr = (uint32_t*)((char*)gpus[i].devmem_addr + gpus[i].sq_cmd_offset);
        gpus[i].grbm_gfx_index_addr = (uint32_t*)((char*)gpus[i].devmem_addr +
                gpus[i].grbm_gfx_index_offset);
    }

    // Catch signals so that we can cleanly quit the application and turn back
    // on any stopped waves. Otherwise, they could die in the CWSR handler.
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);
    signal(SIGQUIT, sig_handler);
    signal(SIGHUP, sig_handler);

    func_arg_t *thread_args;
    thread_args = (func_arg_t*)malloc(num_gpus * sizeof(func_arg_t));

    std::cout << "Beginning work to cause EDP events..." << std::endl;
    std::cout << std::dec;
    std::cout << "    Running for " << loops << " loops..." << std::endl;

    uint32_t interval = (double)loops * 0.1;
    if (interval == 0)
        interval = 1;

    for (uint32_t times_through = 0; times_through < loops; times_through++)
    {
        spinval1.store(0, std::memory_order_seq_cst);
        spinval2.store(0, std::memory_order_seq_cst);

        if (times_through % interval == 0)
            std::cout << times_through+1 << ".. " << std::flush;
        for (uint32_t i = 0; i < num_gpus; i++)
        {
            thread_args[i].which_gpu = i;
            thread_args[i].start_ns = 0;
            thread_args[i].restart_ns = 0;
            thread_args[i].sleep_us = sleep_us;
            pthread_create(&child_threads[i], NULL, pause_and_restart_waves,
                    &thread_args[i]);
        }

        for (uint32_t i = 0; i < num_gpus; i++)
        {
            pthread_join(child_threads[i], NULL);
        }

        if (debug && times_through % interval == 0)
        {
            std::cout << std::endl;
            uint64_t first_start = 0;
            uint64_t last_restart = 0;
            std::cout << std::dec;
            std::cout << "Start and stop time of the wave-restart work on each device:" << std::endl;
            for (uint32_t i = 0; i < num_gpus; i++)
            {
                if (i == 0)
                {
                    first_start = thread_args[i].start_ns;
                    last_restart = thread_args[i].restart_ns;
                }
                else
                {
                    if (thread_args[i].start_ns < first_start)
                        first_start = thread_args[i].start_ns;
                    if (thread_args[i].restart_ns > last_restart)
                        last_restart = thread_args[i].restart_ns;
                }
                std::cout << "\tGPU " << i << ": ";
                std::cout << thread_args[i].start_ns << " -- ";
                std::cout << thread_args[i].restart_ns << " :: (";
                std::cout << (thread_args[i].restart_ns - thread_args[i].start_ns) << " ns)" << std::endl;
            }
            std::cout << "Total time taken to start all the waves: " << (last_restart - first_start) << " ns (";
            std::cout << first_start << " -- " << last_restart << ")" << std::endl;
        }

        usleep(loop_delay_us);
    }
    std::cout << std::endl;

    free(thread_args);
    free(child_threads);
    free(gpus);
    return 0;
}
