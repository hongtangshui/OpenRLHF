#!/usr/bin/env python3
import json
import os
import argparse
from tqdm import tqdm
from openrlhf.utils.matheval.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from openrlhf.utils.matheval.parser import *
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
    with open(input_file, 'r', encoding='utf-8') as f:
        examples = [json.loads(line.strip()) for line in f]

    correct_cnt = 0
    
    # Process each example
    for i, example in enumerate(tqdm(examples, desc="Checking correctness...")):
        gt_cot, gt_ans = parse_ground_truth(example)
        generated_responses = example['generated_responses']
        generated_answers = [extract_answer(response) for response in generated_responses]
        
        # Check correctness
        is_correct_list = [check_is_correct(ans, gt_ans) for ans in generated_answers]
        is_correct = any(is_correct_list)
        
        if is_correct:
            correct_cnt += 1
            
        # Update output data
        example['generated_answers'] = generated_answers
        example['gold_answer'] = gt_ans
        example['is_correct'] = is_correct
        example['answers_correctness'] = is_correct_list

    # Write results directly to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(examples, desc="Writing generations to jsonl file..."):
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
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
    
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
        
    evaluate_generations(args.input_file, args.output_file)

if __name__ == "__main__":
    main()