# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time

_SOLUTION_CLIP_CHARS = 300


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    extract_start = time.time()
    print(f"[GSM8K_EXTRACT_SOLUTION] Starting solution extraction, method: {method}, input length: {len(solution_str)}")

    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        print(f"[GSM8K_EXTRACT_SOLUTION] Clipping solution string from {len(solution_str)} to {_SOLUTION_CLIP_CHARS} characters")
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    if method == "strict":
        print(f"[GSM8K_EXTRACT_SOLUTION] Using strict method with regex pattern")
        regex_start = time.time()
        # this also tests the formatting of the model
        solutions = re.findall("#### (\\-?[0-9\\.\\,]+)", solution_str)
        regex_end = time.time()
        print(f"[GSM8K_EXTRACT_SOLUTION] Regex matching completed in {regex_end - regex_start:.4f} seconds, found {len(solutions)} solutions")
        
        if len(solutions) == 0:
            final_answer = None
            print(f"[GSM8K_EXTRACT_SOLUTION] No solutions found with strict method")
        else:
            # take the last solution
            final_answer = solutions[-1].replace(",", "").replace("$", "")
            print(f"[GSM8K_EXTRACT_SOLUTION] Using last solution: {final_answer}")
    elif method == "flexible":
        print(f"[GSM8K_EXTRACT_SOLUTION] Using flexible method with regex pattern")
        regex_start = time.time()
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        regex_end = time.time()
        print(f"[GSM8K_EXTRACT_SOLUTION] Regex matching completed in {regex_end - regex_start:.4f} seconds, found {len(answer)} numbers")
        
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            print(f"[GSM8K_EXTRACT_SOLUTION] No numbers found with flexible method")
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    print(f"[GSM8K_EXTRACT_SOLUTION] Using valid number: {final_answer}")
                    break
    
    extract_end = time.time()
    total_extract_time = extract_end - extract_start
    print(f"[GSM8K_EXTRACT_SOLUTION] Solution extraction completed in {total_extract_time:.4f} seconds, result: {final_answer}")
    
    return final_answer


def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual
    Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    start_time = time.time()
    print(f"[GSM8K_COMPUTE_SCORE] Starting GSM8K score computation at {time.strftime('%H:%M:%S')}")
    print(f"[GSM8K_COMPUTE_SCORE] Method: {method}, solution_str length: {len(solution_str)}")
    
    extract_start = time.time()
    answer = extract_solution(solution_str=solution_str, method=method)
    extract_end = time.time()
    print(f"[GSM8K_COMPUTE_SCORE] Solution extraction completed in {extract_end - extract_start:.4f} seconds")
    
    if answer is None:
        print(f"[GSM8K_COMPUTE_SCORE] No answer extracted, returning 0")
        result = 0
    else:
        print(f"[GSM8K_COMPUTE_SCORE] Extracted answer: {answer}, ground_truth: {ground_truth}")
        if answer == ground_truth:
            print(f"[GSM8K_COMPUTE_SCORE] Answer matches ground truth, returning {score}")
            result = score
        else:
            print(f"[GSM8K_COMPUTE_SCORE] Answer does not match ground truth, returning {format_score}")
            result = format_score
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"[GSM8K_COMPUTE_SCORE] Total GSM8K computation time: {total_time:.4f} seconds")
    print(f"[GSM8K_COMPUTE_SCORE] GSM8K score computation completed at {time.strftime('%H:%M:%S')}")
    
    return result
