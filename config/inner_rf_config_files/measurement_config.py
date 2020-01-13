"""The measurement configuration file consists of the parameters required when inferring with point cloud data obtained from measurement systems
                
        :param ms_parameters['measurement_files']: List of measurement files obtained from the measurement system, currently the model is configured to process data (tab delimited output file with features and surface points) from WLS400 Hexagon 3D Optical scanner refer: https://www.hexagonmi.com/products/white-light-scanner-systems/hexagon-metrology-wls400a fro more details
        :type ms_parameters['measurement_files']: list (required)
"""

ms_parameters = {	
        'measurement_files':['MC1.txt','MC2.txt']
        }