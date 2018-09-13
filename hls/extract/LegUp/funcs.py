# -*- coding: utf-8 -*-
import os
import re


def Make(ns, log_file):
    # remove all CP folder
    # os.system('rm -r ./*CP_*')
    
    for n in ns:
        # midify config file
        Make_modify_config(n, log_file)
    
        # do make
        try:
            os.system('make')
            pass
        except:
            log_file.write('Make Failed! ' + os.path.abspath('.') + ' with clock period: ' + str(n) + '\n')
    
        # do make p
        try:
            os.system('make p')
            pass
        except:
            log_file.write('Make P Failed! ' + os.path.abspath('.') + ' with clock period: ' + str(n) + '\n')
        
        # copy files
        design_name = os.path.abspath('.').split('/')[-1]
        folder_name = './' + design_name + '_CP_' + str(n)
        os.system('mkdir ' + folder_name)
        os.system('cp ./* ' + folder_name)
        
        # clean
        os.system('make clean')
        os.system('make cleanall')
        
        # succeed
        log_file.write('Make Succeed! ' + os.path.abspath('.') + ' with clock period: ' + str(n) + '\n')


def Make_modify_config(n, log_file, file_name='./config.tcl'):
    print 'Modify config file -', os.path.abspath(file_name)
        
    if not os.path.exists(file_name):
        config = open(file_name, 'w')
        config.write('set_parameter CLOCK_PERIOD '+str(n))
        config.close()
    else:
        data = ''
        with open(file_name, 'r') as f:
            for line in f.readlines():
                if line.find('set_parameter CLOCK_PERIOD') > -1:
                    data += ''
                else:
                    data += line
            data += 'set_parameter CLOCK_PERIOD ' + str(n)
        
        with open(file_name, 'w') as f:
            f.writelines(data)
        
    log_file.write('Modify Config File Succeed!' + os.path.abspath('.') + '\n')
    return True


def MakeF(log_file):
    for y in os.listdir('.'):
#        if re.search('^.*?CP_[0-9]+$', y):
        if re.search('^.*?(CP_1$)|(CP_15$)', y):
            os.chdir(y)
            
            if MakeF_modify_qsf(log_file):
                MakeF_run_command(log_file)
                
            os.chdir('..')
        

def MakeF_run_command(log_file):
    print 'Run Command -', os.path.abspath('.')
    
    try:
        os.system('quartus_sh "--64bit" --flow compile top')
    except:
        log_file.write('Run Command Failed!' + os.path.abspath('.') + '\n')
        return False
        
    log_file.write('Run Command Succeed!' + os.path.abspath('.') + '\n')
    return True


def MakeF_modify_qsf(log_file, file_name='./top.qsf'):
    print 'Modify file -', os.path.abspath(file_name)
        
    if not os.path.exists(file_name):
        log_file.write('Modify QSF File Failed! Cannot find QSF File ' + os.path.abspath('.') + '\n')
        return False
    else:
        data = ''
        with open(file_name, 'r') as f:
            for line in f.readlines():
                if line.find('source /common/legup-4.0/boards/common.qsf') > -1:
                    data += ""
                elif line.find('set_global_assignment -name SEARCH_PATH "/common/legup-4.0/tiger/processor/altera_libs') > -1:
                    data += ""
                else:
                    data += line

            # data += '\nset_global_assignment -name FAMILY "Arria 10"'

        with open(file_name, 'w') as f:
            f.writelines(data)
        
    log_file.write('Modify Config File Succeed!' + os.path.abspath('.') + '\n')
    return True


def CheckFiles(log_file):
    for y in os.listdir('.'):
        if re.match('^.*?CP_[0-9]+$', y):
            os.chdir(y)
            
            # check files
            find_res = os.path.exists('resources.legup.rpt')
            find_fit = os.path.exists('top.fit.rpt')
            find_sch = os.path.exists('scheduling.legup.rpt')
            info_cp = ""
            
            # get info
            if find_sch:
                with open('scheduling.legup.rpt', 'r') as f:
                    for line in f.readlines():
                        if line.find('Clock period constraint:') > -1:
                            info_cp = line[0:-1]
                            break
                        
            # print and record
            info = os.path.abspath('.') + ',' + info_cp
            
            if find_res: info += ',resources.legup.rpt'
            else: info += ','
            
            if find_fit: info += ',top.fit.rpt'
            else: info += ','
            
            print info
            log_file.write(info + '\n')
            
            # if find_sch: info += ','
            # else: info += ','
            
            os.chdir('..')
    
    
Feature1_Names = ['Registers',
                  'DSP Elements',
                  'Combinational',
                  'RAM Elements',
                  'Logic Elements',
                  'Clock Period',
                  'Delay_of_path_max',
                  'Delay_of_path_min',
                  'Delay_of_path_mean',
                  'Delay_of_path_med']

Feature2_Names = ['signed_add_32',
                  'signed_add_64',
                  'signed_comp_eq_32',
                  'signed_comp_eq_64',
                  'signed_multiply_32',
                  'signed_comp_eq_mux_32',
                  'signed_subtract_32',
                  'signed_add_8',
                  'signed_comp_eq_8',
                  'signed_comp_lt_8',
                  'unsigned_comp_lt_8',
                  'shift_ll_32',
                  'shift_rl_32',
                  'altfp_extend_32',
                  'altfp_subtract_32',
                  'altfp_subtract_64',
                  'signed_comp_ogt_64',
                  'signed_comp_olt_64',
                  'altfp_add_32',
                  'altfp_add_64',
                  'signed_comp_oeq_32',
                  'signed_comp_oeq_64',
                  'altfp_multiply_32',
                  'altfp_multiply_64',
                  'altfp_divide_64',
                  'altfp_sitofp_64',
                  'altfp_fptosi_32',
                  'altfp_divide_32',
                  'signed_comp_ogt_32',
                  'signed_comp_olt_32',
                  'signed_comp_ugt_32',
                  'signed_comp_une_32',
                  'signed_comp_gt_32',
                  'signed_comp_lt_32',
                  'altfp_sitofp_32',
                  'signed_divide_32',
                  'unsigned_divide_32',
                  'bitwise_AND_32',
                  'bitwise_OR_32',
                  'bitwise_XOR_32',
                  'unsigned_comp_gt_32',
                  'bitwise_AND_64',
                  'bitwise_OR_64',
                  'shift_ll_64',
                  'shift_rl_64',
                  'signed_comp_eq_mux_64',
                  'signed_comp_lt_64',
                  'signed_subtract_64',
                  'unsigned_comp_lt_32',
                  'unsigned_comp_lt_64',
                  'shift_ra_32',
                  'signed_multiply_64',
                  'signed_modulus_32',
                  'bitwise_XOR_64',
                  'unsigned_comp_gt_64',
                  'unsigned_divide_64',
                  'bitwise_OR_16',
                  'signed_add_16',
                  'signed_comp_eq_16',
                  'signed_comp_eq_mux_16',
                  'signed_comp_gt_16',
                  'signed_comp_gte_32',
                  'signed_comp_lt_16',
                  'signed_multiply_nodsp_32',
                  'signed_subtract_16',
                  'signed_comp_gt_8',
                  'signed_comp_gt_64',
                  'bitwise_XOR_8',
                  'altfp_fptosi_64',
                  'signed_comp_ole_32',
                  'signed_comp_ult_32',
                  'signed_comp_eq_mux_8',
                  'signed_comp_lte_32',
                  'unsigned_comp_lte_64',
                  'signed_comp_lte_64',
                  'unsigned_comp_lt_16',
                  'signed_multiply_nodsp_64',
                  'shift_ra_64',
                  'signed_divide_64',
                  'unsigned_modulus_32']

Target_Names = ['Registers_used',
                'DSP_blocks_used',
#               'ALMS_used',
#               'Total_ALMS',
#               'Total_DSP_blocks',
#               'ALUT_for_logic',
#               'ALUT_for_route-throughs',
#               'ALUT_for_memory',
                'ALUT_used',
#               'Pins_used',
#               'Total_Pins',
#               'Virtual_pins_used',
                'RAM_blocks_used'
                'Block_memory_bits_used']
#               'Total_Block_memory_bits',
#               'Total_RAM_blocks',
#               'HSSI_RX_channles_used',
#               'Total_HSSI_RX_channles',
#               'HSSI_TX_channles_used',
#               'Total_HSSI_TX_channles',
#               'PLLs_used',
#               'Total_PLLs']      


def ExtractData(design_index, data_file):
    global Feature1_Names, Feature2_Names, Target_Names
        
    if data_file.tell() == 0:
        head = "Design_Path,Design_Index,Device_Index"
        for name in Feature1_Names:
            head += ',' + name
        for name in Feature2_Names:
            head += ',' + name
        for name in Target_Names:
            head += ',' + name
        data_file.write(head + '\n')
        
    for y in os.listdir('.'):
        if re.match('^.*?CP_[0-9]+$', y):
            os.chdir(y)
            ExtractData_file(design_index, data_file)
            os.chdir('..')
            
    # DataFile.close()
    
    
def ExtractData_file(design_index, data_file):
    global Feature1_Names, Feature2_Names, Target_Names
    
#    # design name
#    design_name = os.path.abspath('.').split('/')[-2]
    
    # .v file
    vfs = []
    for path in os.listdir('.'):
        # os.path.splitext():分离文件名与扩展名
        if os.path.splitext(path)[1] == '.v':
            vfs.append(path)
    
    # get info
    result = {}
    for name in Feature1_Names:
        result[name] = ''
    for name in Feature2_Names:
        result[name] = 0
    for name in Target_Names:
        result[name] = ''
    
    file_name = 'scheduling.legup.rpt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f.readlines():
                
                if line.find('Clock period constraint') > -1:
                    data = re.search(': (.+)ns', line).group(1)
                    result['Clock Period'] = float(data)
                    break
    
    file_name = 'resources.legup.rpt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f.readlines():
                
                for name in ['Logic Elements', 'Combinational', 'Registers', 'DSP Elements']:
                    if line.find(name) > -1:
                        data = re.search(': (.+)$', line)
                        result[name] = int(data.group(1))
                        
                for name in Feature2_Names:
                    if line.find('Operation "' + name + '" x ') > -1:
                        data = re.search('Operation ".+" x ([0-9,]+)', line)
                        result[name] = int(data.group(1).replace(',', ''))
            
    file_name = 'timingReport.legup.rpt'
    if os.path.exists(file_name):
        f_dops = []
        with open(file_name, 'r') as f:
            for line in f.readlines():
                
                if line.find('-----------------Delay of path:') > -1:
                    data = re.search('-Delay of path:([0-9,.]+) ns-', line)
                    if data is not None:
                        f_dops.append(float(data.group(1).replace(',', '')))
                        
        # calculated data
        if len(f_dops) > 0:
            import numpy as np
            result['Delay_of_path_max'] = np.max(f_dops)
            result['Delay_of_path_min'] = np.min(f_dops)
            result['Delay_of_path_mean'] = np.mean(f_dops)
            result['Delay_of_path_med'] = np.median(f_dops)
        else:
            result['Delay_of_path_max'] = 0
            result['Delay_of_path_min'] = 0
            result['Delay_of_path_mean'] = 0
            result['Delay_of_path_med'] = 0
    
    for file_name in vfs:
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                for line in f.readlines():
                    
                    if line.find('// Number of RAM elements:') > -1:
                        data = re.search('// Number of RAM elements: ([0-9,]+)', line)
                        if data is not None:
                            result['RAM Elements'] = int(data.group(1).replace(',', ''))
                            
#    file_name = 'memory.legup.rpt'
#    if os.path.exists(file_name):
#        f_rme = 0
#        with open(file_name, 'r') as f:
#            for line in f.readlines():
#                
#                if line.find('ram: ') > -1:
#                    data = re.search('ram:.+size.+alignment.+offset.+unused.+', line)
#                    if data is not None:
#                        f_rme += 1   
#                        
#        # calculated data
#        result['RAM Elements'] = f_rme
     
    file_name = 'top.fit.rpt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f.readlines():
                
                if line.find('; Total registers') > -1:
                    data = re.search('; ([0-9,]+) ', line)
                    if data is not None:
                        result['Registers_used'] = int(data.group(1).replace(',', ''))
                        
                if line.find('; Total block memory bits') > -1:
                    data = re.search('; ([0-9,]+) / ([0-9,]+)', line)
                    if data is not None:
                        result['Block_memory_bits_used'] = int(data.group(1).replace(',', ''))
                        result['Total_Block_memory_bits'] = int(data.group(2).replace(',', ''))
                        
                if line.find('; Total RAM Blocks') > -1:
                    data = re.search('; ([0-9,]+) / ([0-9,]+)', line)
                    if data is not None:
                        result['RAM_blocks_used'] = int(data.group(1).replace(',', ''))
                        result['Total_RAM_blocks'] = int(data.group(2).replace(',', ''))
                        
                if line.find('; Total DSP Blocks') > -1:
                    data = re.search('; ([0-9,]+) / ([0-9,]+)', line)
                    if data is not None:
                        result['DSP_blocks_used'] = int(data.group(1).replace(',', ''))
                        result['Total_DSP_blocks'] = int(data.group(2).replace(',', ''))
                        
                if line.find('; Combinational ALUT usage for logic') > -1:
                    data = re.search('; ([0-9,]+) ', line)
                    if data is not None:
                        result['ALUT_for_logic'] = int(data.group(1).replace(',', ''))
                        
                if line.find('; Combinational ALUT usage for route-throughs') > -1:
                    data = re.search('; ([0-9,]+) ', line)
                    if data is not None:
                        result['ALUT_for_route-throughs'] = int(data.group(1).replace(',', ''))
                        
                if line.find('; Memory ALUT usage') > -1:
                    data = re.search('; ([0-9,]+) ', line)
                    if data is not None:
                        result['ALUT_for_memory'] = int(data.group(1).replace(',', ''))
            
        # calculated data
        if result.has_key('ALUT_for_logic') or result.has_key('ALUT_for_route-throughs') or result.has_key('ALUT_for_memory'):
            if not result.has_key('ALUT_for_logic'): result['ALUT_for_logic'] = 0
            if not result.has_key('ALUT_for_route-throughs'): result['ALUT_for_route-throughs'] = 0
            if not result.has_key('ALUT_for_memory'): result['ALUT_for_memory'] = 0
            result['ALUT_used'] = result['ALUT_for_logic'] + result['ALUT_for_route-throughs'] + result['ALUT_for_memory']
                
    # write file
    if result['Registers_used'] != '' and result['DSP_blocks_used'] != '':
        # output string
        output = os.path.abspath('.') + ',' + str(design_index) + ',0'
        for name in Feature1_Names:
            output += ',' + str(result[name])
        for name in Feature2_Names:
            output += ',' + str(result[name])
        for name in Target_Names:
            output += ',' + str(result[name])
        
        data_file.write(output + '\n')
        print output
        

What_Features = []

def WhatFeatures():
    global What_Features
        
    for y in os.listdir('.'):
        if re.match('^.*?CP_[0-9]+$', y):
            os.chdir(y)
            WhatFeatures_features()
            os.chdir('..')
            
           
def WhatFeatures_features():
    file_name = 'resources.legup.rpt'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            for line in f.readlines():
                if line.find('Operation "') > -1:
                    data = re.search('Operation "(.+)"', line).group(1)
                    if data not in What_Features:
                        What_Features.append(data)
                        print data
            
            
if __name__ == '__main__':
    pass
#    ExtractData()

    
# more CP: 15, 1
    