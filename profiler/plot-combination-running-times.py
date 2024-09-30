import pandas as pd
import matplotlib.pyplot as plt

import sys

# Function to read the log file and process data
def read_log_file(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Parse relevant information from the log file
            if 'Running combination' in line:
                timestamp = line.split(' - ')[0]
                try:
                    details_str = line.split('{')[-1].strip()
                    if details_str.endswith('}'):
                        details = eval('{'+details_str)  # Extract the dictionary part of the line
                        process_batch_size = details['process_batch_size']
                        arena_batch_size = details['arena_batch_size']
                        numMCTSSims = details['numMCTSSims']
                        data.append([timestamp, process_batch_size, arena_batch_size, numMCTSSims])
                    else:
                        raise SyntaxError("Malformed dictionary string")
                except (SyntaxError, NameError) as e:
                    print(f"Skipping line due to error: {e} - {line.strip()}")
    
    # Convert the list to a DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'process_batch_size', 'arena_batch_size', 'numMCTSSims'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate running times in seconds as the difference between the current and next timestamp
    df['running_time'] = (df['timestamp'].shift(-1) - df['timestamp']).dt.total_seconds()
    
    # Drop the last row as it will have NaN running time
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python full-plot-from-web-from-file.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    df = read_log_file(log_file)
    print(df)

    # Display the full table
    df_table = df[['process_batch_size', 'arena_batch_size', 'numMCTSSims', 'running_time']]
    print(df_table)

    # Improved plot with each process and arena batch size combination as a different color/series

    plt.figure(figsize=(10, 6))

    # Define different markers and colors for clarity
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', '|', '_']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # Plot 1: Running time vs numMCTSSims with enhanced readability and distinct series
    for idx, ((pbs, abs_), marker, color) in enumerate(zip(df.groupby(['process_batch_size', 'arena_batch_size']).groups.keys(), markers, colors)):
        subset = df[(df['process_batch_size'] == pbs) & (df['arena_batch_size'] == abs_)]
        plt.plot(subset['numMCTSSims'], subset['running_time'], marker=marker, color=color, linestyle='-', linewidth=2, label=f'process_batch_size = {pbs}, arena_batch_size = {abs_}', markersize=8)

    plt.title('Running Time vs numMCTSSims', fontsize=16)
    plt.xlabel('numMCTSSims', fontsize=14)
    plt.ylabel('Running Time (seconds)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=12, loc='upper left')

    plt.tight_layout()
    plt.show()
