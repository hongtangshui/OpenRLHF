#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm
from openrlhf.utils.matheval.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from openrlhf.utils.matheval.parser import *
from openrlhf.utils.matheval.data_loader import load_data
from openrlhf.utils.matheval.math_normalization import *
from openrlhf.utils.matheval.grader import *

def evaluate_generations(input_file, output_file):
    """
    Evaluate generated responses and save results to output file.
    
    Args:
        input_file (str): Path to input jsonl file
        output_file (str): Path to output jsonl file
    """
    # Read input file
    examples = []
    file_outputs = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            examples.append(data)
            file_outputs.append(data)

    correct_cnt = 0
    
    # Process each example
    for i in tqdm(range(len(examples)), desc="Checking correctness..."):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d)
        generated_responses = file_outputs[i]['generated_responses']
        generated_answers = [extract_answer(generated_response) 
                           for generated_response in generated_responses]
        
        # Check correctness
        is_correct_list = [check_is_correct(generated_answer, gt_ans) 
                          for generated_answer in generated_answers]
        is_correct = any(is_correct_list)
        
        if is_correct:
            correct_cnt += 1
            
        # Update output data
        file_outputs[i]['generated_answers'] = generated_answers
        file_outputs[i]['gold_answer'] = gt_ans
        file_outputs[i]['is_correct'] = is_correct
        file_outputs[i]['answers_correctness'] = is_correct_list

    # Write results to temporary file first
    temp_out_file = output_file + ".tmp"
    try:
        with open(temp_out_file, 'w', encoding='utf-8') as f:
            for d in tqdm(file_outputs, desc="Writing generations to jsonl file..."):
                f.write(json.dumps(d, ensure_ascii=False))
                f.write("\n")
                if i % 100 == 0:
                    f.flush()
            f.flush()
        
        # Rename temporary file to final output file
        os.replace(temp_out_file, output_file)
    except Exception as e:
        # Clean up temp file if something goes wrong
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        raise e
    
    # Print statistics
    print(f"Correct count / total count: {correct_cnt}/{len(examples)}")
    print(f"Accuracy: {correct_cnt / len(examples):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate generated responses')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to input jsonl file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to output jsonl file')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    evaluate_generations(args.input_file, args.output_file)


if __name__ == "__main__":
    main()