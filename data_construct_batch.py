import requests
import json
import time
from datetime import datetime
from tqdm import tqdm

def generate_scenario(domain: str, level: int, api_key: str) -> dict:
    """
    Generate a single visualization scenario
    """
    url = "https://gptproxy.llmpaas.tencent.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": f"""Generate a visualization scenario for {domain} domain at reasoning level {level}.
            Create a realistic analysis scenario that includes:
            1. Initial data context
            2. A sequence of 4 progressive visualization queries
            3. Expected visualization types
            4. Key insights to be discovered

            Format the output as a JSON object with these fields:
            {{
                "domain": "{domain}",
                "level": {level},
                "context": "description of the context",
                "queries": ["query1", "query2", "query3", "query4"],
                "vis_types": ["type1", "type2", "type3", "type4"],
                "insights": ["insight1", "insight2", ...]
            }}"""
        }],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                try:
                    scenario = json.loads(content)
                    return scenario
                except json.JSONDecodeError:
                    print(f"\nWarning: Could not parse scenario as JSON for {domain} level {level}")
                    return None
        else:
            print(f"\nError: API request failed with status code {response.status_code}")
            return None

    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return None

def generate_batch_scenarios(api_key: str, batch_size: int = 10):
    """
    Generate multiple scenarios and save them to a file
    """
    # Test domains and levels
    domains = ["business", "healthcare", "education", "finance", "retail"]
    levels = [1, 2, 3, 4]
    
    # Create timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"vis_scenarios_{timestamp}.json"
    log_filename = f"vis_scenarios_log_{timestamp}.txt"
    
    scenarios = []
    failed_attempts = []
    
    # Initialize progress bar
    total_scenarios = batch_size
    with tqdm(total=total_scenarios) as pbar:
        for i in range(total_scenarios):
            # Cycle through domains and levels
            domain = domains[i % len(domains)]
            level = levels[i % len(levels)]
            
            # Generate scenario
            scenario = generate_scenario(domain, level, api_key)
            
            if scenario:
                scenarios.append(scenario)
                # Save after each successful generation (incremental saving)
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(scenarios, f, indent=2, ensure_ascii=False)
            else:
                failed_attempts.append({"domain": domain, "level": level, "index": i})
            
            # Update progress bar
            pbar.update(1)
            
            # Add delay to prevent rate limiting
            time.sleep(2)
            
    # Save final results
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(scenarios, f, indent=2, ensure_ascii=False)
    
    # Save log with failed attempts
    with open(log_filename, 'w', encoding='utf-8') as f:
        f.write(f"Total scenarios attempted: {total_scenarios}\n")
        f.write(f"Successful scenarios: {len(scenarios)}\n")
        f.write(f"Failed attempts: {len(failed_attempts)}\n\n")
        f.write("Failed scenarios details:\n")
        json.dump(failed_attempts, f, indent=2)
    
    print(f"\nGeneration completed!")
    print(f"Successfully generated scenarios: {len(scenarios)}")
    print(f"Failed attempts: {len(failed_attempts)}")
    print(f"Results saved to: {filename}")
    print(f"Log saved to: {log_filename}")
    
    return scenarios

def test_single_scenario(api_key: str):
    """
    Test generating a single scenario
    """
    print("=== Testing Single Scenario ===")
    scenario = generate_scenario("business", 2, api_key)
    if scenario:
        print("\nGenerated Scenario:")
        print(json.dumps(scenario, indent=2))
    return scenario

if __name__ == "__main__":
    # Your API key
    api_key = "8beCPow2KcmVGufSecmUZrTQhVN2OnPb"  # Replace with your key
    
    # Ask user for mode
    mode = input("Choose mode (1 for single test, 2 for batch generation): ")
    
    if mode == "1":
        test_single_scenario(api_key)
    elif mode == "2":
        batch_size = int(input("Enter number of scenarios to generate: "))
        generate_batch_scenarios(api_key, batch_size)
    else:
        print("Invalid mode selected")