from openai import OpenAI
import subprocess
client = OpenAI(api_key="insert_key")


def get_input_from_user(prompt):
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )

        # ✅ Correct way to extract the text from GPT response
        gpt_response = response.choices[0].message.content.strip()  

        try:
            return int(gpt_response)  # Convert response to integer
        except ValueError:
            print(f"GPT-4o asked: {gpt_response}")  # Show GPT's response
            prompt = input("Your response: ")


def interact_with_user():
    # Start the conversation with GPT
    print("Hi! Let's figure out the parameters for the nurse rostering problem.")
    
    # Prepare the initial prompt to ask GPT to start the conversation and ask for parameters
    initial_prompt = "You are a helpful assistant. Your task is to gather the parameters for a nurse rostering problem. Start the conversation by asking the user for the number of nurses, shifts per day, and the number of days. Then, ask clarifying questions if needed, but don't hardcode the questions."
    
    # First input from GPT
    user_input = get_input_from_user(initial_prompt)
    print(user_input)  # Display GPT's response to the user
    
    # Continue asking questions until the agent gathers all the required parameters
    num_nurses = None
    num_shifts = None
    num_days = None

    # Start a loop for getting answers
    while num_nurses is None or num_shifts is None or num_days is None:
        user_answer = input("Your response: ")
        
        # Send user's answer back to GPT for the next part of the conversation
        if num_nurses is None:
            prompt = f"User said: {user_answer}. Based on this, ask the user for the number of nurses."
            num_nurses = int(get_input_from_user(prompt))
        elif num_shifts is None:
            prompt = f"User said: {user_answer}. Based on this, ask the user for the number of shifts per day."
            num_shifts = int(get_input_from_user(prompt))
        elif num_days is None:
            prompt = f"User said: {user_answer}. Based on this, ask the user for the number of days."
            num_days = int(get_input_from_user(prompt))
        
        # Make sure GPT knows if it needs clarification
        user_answer = input("Your response: ")
        
    print(f"Got all the parameters: {num_nurses}, {num_shifts}, {num_days}")
    
    # After collecting the parameters, run the solver
    run_solver(num_nurses, num_shifts, num_days)

def run_solver(num_nurses, num_shifts, num_days):
    # Call the solver program by passing the parameters as arguments
    subprocess.run(['python', '../src/nrp_or-tools.py', str(num_nurses), str(num_shifts), str(num_days)])

if __name__ == "__main__":
    interact_with_user()
