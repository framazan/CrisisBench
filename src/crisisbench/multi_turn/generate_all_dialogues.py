# Generate all dialogues using all config files in a given dialogue_configs directory 

import os
import argparse
from generate_dialogue import generate_dialogue, load_config

def process_all_dialogue_configs(input_dir, output_dir):
    """
    Processes all dialogue configuration files in the specified input directory
    by calling the generate_dialogue function on each and saving the output
    to the specified output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the specified input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".yaml"):  # Assuming all config files are YAML
            input_file_path = os.path.join(input_dir, filename)
            output_file_path = os.path.join(output_dir, filename)
            print(f"Generating dialogue for {input_file_path} and saving to {output_file_path}")
            
            # Load the configuration from the file
            config = load_config(input_file_path)
            
            print(f"Loaded configuration: {config}")
            print(f"Output directory: {output_dir}")
            # Generate dialogue using the loaded configuration
            generate_dialogue(config, output_dir)
        else:
            raise ValueError(f"Directory {input_dir} contains non-YAML files that are not dialogue configs")

def main():
    parser = argparse.ArgumentParser(description="Process dialogue configuration files.")
    parser.add_argument('--input_dir', required=True, help='Directory containing dialogue config files')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated dialogues')

    args = parser.parse_args()

    print(f"Processing all dialogue configs in {args.input_dir} and saving to {args.output_dir}")
    process_all_dialogue_configs(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()

