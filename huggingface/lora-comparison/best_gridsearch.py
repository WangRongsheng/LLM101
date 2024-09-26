def find_top_accuracies(file_path, top_n=5):
    with open(file_path, 'r') as file:
        log_data = file.read()

    # Splitting the log into sections
    sections = log_data.split('------------------------------------------------\n------------------------------------------------\n')[1:]

    # Extracting test accuracies and their respective sections
    acc_section_pairs = []

    for section in sections:
        # Find the line with the test accuracy
        for line in section.split('\n'):
            if line.startswith("Test acc:"):
                # Extracting the accuracy value
                acc = float(line.split()[-1].replace('%', ''))
                acc_section_pairs.append((acc, section))
                break

    acc_section_pairs.sort(key=lambda x: x[0], reverse=True)
    top_entries = acc_section_pairs[:top_n]

    return top_entries

file_path = 'results.txt'
top_5_entries = find_top_accuracies(file_path)

# Printing the top 5 entries
for i, entry in enumerate(top_5_entries, 1):
    accuracy, section = entry
    print(f"Top {i}: Test Accuracy: {accuracy}%\n{section}\n")