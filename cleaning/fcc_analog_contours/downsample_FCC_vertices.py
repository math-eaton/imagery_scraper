def downsample_polar_coordinates(input_filename, output_filename, limit=20):
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        # Read the first line (header) and identify relevant columns
        headers = infile.readline().strip().split('|')
        transmitter_site_index = headers.index('transmitter_site')
        end_index = headers.index('^')
        
        # Calculate the indices of the headers for downsampled coordinates
        downsampled_indices = list(range(transmitter_site_index + 1, end_index, 3))[:120]
        downsampled_headers = [headers[i] for i in downsampled_indices]
        
        # Write a modified header to the output file
        selected_headers = headers[:transmitter_site_index + 1] + \
                           downsampled_headers + \
                           [headers[end_index]]
        outfile.write('|'.join(selected_headers) + '\n')
        
        # Initialize a counter for processed records
        record_count = 0
        
        # Process each record in the file, up to the limit
        for line in infile:
            if record_count >= limit:
                break  # Stop processing if the limit is reached
            
            fields = line.strip().split('|')
            transmitter_site = fields[transmitter_site_index]
            polar_coordinates = fields[transmitter_site_index + 1:end_index]
            # Select every 3rd coordinate, up to the first 120
            downsampled_coordinates = [polar_coordinates[i] for i in range(0, len(polar_coordinates), 3)][:120]
            
            # Construct the new line and write to the output file
            new_line = '|'.join([transmitter_site] + downsampled_coordinates + ['^'])
            outfile.write(new_line + '\n')
            
            record_count += 1  # Increment the counter



# Example usage
input_filename = 'data/raw/fm_contour_minSample.txt'
output_filename = 'data/processed/fm_contours_20240307/downsampled.txt'
downsample_polar_coordinates(input_filename, output_filename)
