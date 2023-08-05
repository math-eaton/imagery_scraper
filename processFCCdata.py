import csv

with open('data/FM_service_contour_current.csv', 'r') as source_file, open('data/FM_service_contour_processed.csv', 'w', newline='') as target_file:
    reader = csv.reader(source_file, delimiter=',')
    writer = csv.writer(target_file, delimiter=',')

    headers = next(reader)
    new_headers = headers[:5] + ['latitude'+str(i) for i in range(1,362)] + ['longitude'+str(i) for i in range(1,362)]
    writer.writerow(new_headers)

    for index, row in enumerate(reader, start=1):
        new_row = row[:5]
        for i, coord in enumerate(row[5:-1], start=1): # Exclude the last field
            try:
                lat, lon = coord.split(',')
                new_row.append(lat)
                new_row.append(lon)
            except ValueError:
                print(f'Error at row {index}, coord {i}: {coord}')
                new_row.append('')
                new_row.append('')
        writer.writerow(new_row)
        print(f'Processed row {index}')
