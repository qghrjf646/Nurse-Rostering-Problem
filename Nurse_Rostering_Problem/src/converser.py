import openai
import json
import re

def extract_json(reply):
    # Use regex to find the first { ... } block in the reply
    match = re.search(r'\{.*\}', reply, re.DOTALL)
    if match:
        return match.group(0)
    else:
        raise ValueError("No JSON object found in the reply.")

def collect_nurse_rostering_parameters():
    # System message: instruct the assistant on its role and output format.
    system_message = {
        "role": "system",
        "content": (
        "You are a helpful assistant for the task of scheduling nurse shifts. "
        "Your goal is to have a natural conversation with the user to collect all required parameters. "
        "The mandatory parameters are: "
        "  - num_days (number of days of planning), "
        "  - num_nurses (number of nurses), "
        "  - max_consecutive_days (maximum consecutive work days), and "
        "  - num_senior_nurses (number of available senior nurses). "
        "Additionally, optionally ask for preferred constraints: "
        "  - preferred_day_offs: a matrix that is a 2D array [nurse][day] = 1 if a nurse requests that day off, else 0, "
        "  - preferred_shifts: a matrix that is a 3D array [nurse][day][shift] = 1 if a nurse requests that shift on that day, else 0. "
        "If the user declines to provide the optional matrices, output them as null. "
        "If the user mentions nurse names and preferences (for example, 'Stephany would like to work the 2nd shift on the 1st Monday'), "
        "use your parsing skills to map names to indices and set the corresponding entries in the matrices accordingly. "
        "Do not include any inline comments or extra text in your output. "
        "When all parameters have been collected, output a final message starting with the token 'PARAMETERS:' followed by a valid JSON object with exactly these keys: "
        "num_days, num_nurses, max_consecutive_days, num_senior_nurses, preferred_day_offs, preferred_shifts. "
        "Ensure the JSON object contains no extra text or comments."
        )
    }

    messages = [system_message]

    # Set your API key
    openai.api_key = "insert_your_api_key_here"
    # Optionally, if you want to use a custom base URL for another provider, uncomment and set:
    # openai.api_base = "https://api.mini.text-generation-webui.myia.io/v1"

    print("Please answer the assistant's questions as naturally as possible.\n")
    
    # Conversation loop.
    while True:
        user_input = input("User: ")
        messages.append({"role": "user", "content": user_input})

        # Call the ChatCompletion API
        completion = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or another model of your choice
            messages=messages,
            temperature=0.7
        )

        # Extract the assistant's reply
        try:
            reply = completion.choices[0].message["content"]
        except Exception as e:
            print("Error extracting reply:", e)
            continue

        print("Assistant:", reply)
        messages.append({"role": "assistant", "content": reply})

        # If the assistant signals that all parameters are collected with "PARAMETERS:" prefix, break the loop.
        if "PARAMETERS:" in reply.strip():
            break

    # Extract the JSON part.
    # Expecting a reply like: PARAMETERS: { "num_days": 7, "num_nurses": 10, ... }
    try:
        #json_str = extract_json(reply)
        json_str = reply.split("PARAMETERS:", 1)[1].strip()
        params = json.loads(json_str)
    except Exception as e:
        print("Error parsing JSON from the assistant's output:", e)
        params = {}

    return params

if __name__ == "__main__":
    result = collect_nurse_rostering_parameters()
    print("\nCollected Parameters:")
    print(result)