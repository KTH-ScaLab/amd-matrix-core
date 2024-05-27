#include <hip/hip_runtime.h>
#include "rocm_smi/rocm_smi.h"

#include <iostream>
#include <stdint.h>
#include <chrono>
#include <unistd.h>

int main(int argc, char* argv[]) {

    if(argc != 2) {
        std::cerr << "usage: " << argv[0] << " sample_period_ms" << std::endl;
        exit(1);
    }
    int sample_period_ms = std::stoi(argv[1]);

    rsmi_status_t ret;
    uint32_t num_devices;
    uint64_t power;
    uint16_t dev_id;
 
    ret = rsmi_init(0);
    if(ret) {
        std::cerr << "error in rsmi_init" << std::endl;
        exit(1);
    }

    ret = rsmi_num_monitor_devices(&num_devices);
    
    std::chrono::milliseconds ms;
    
    while(true) {
        ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        );
        rsmi_dev_power_ave_get(/* dv_ind */ 0, 0, &power);

        std::cout.precision(3); 
        std::cout 
            << std::fixed << (double) ms.count() / 1000 << ","
            << (double) power / 1e6
            << std::endl;

        usleep(sample_period_ms * 1000);
    }

    ret = rsmi_shut_down();
    return 0;
}
