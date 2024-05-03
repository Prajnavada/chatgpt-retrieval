from langchain.prompts import PromptTemplate

# reasoning_modules = [
#     "1. How could I devise an experiment to help solve that problem?",
#     "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
#     # "3. How could I measure progress on this problem?",
#     "4. How can I simplify the problem so that it is easier to solve?",
#     "5. What are the key assumptions underlying this problem?",
#     "6. What are the potential risks and drawbacks of each solution?",
#     "7. What are the alternative perspectives or viewpoints on this problem?",
#     "8. What are the long-term implications of this problem and its solutions?",
#     "9. How can I break down this problem into smaller, more manageable parts?",
#     "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
#     "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
#     # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
#     "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
#     "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
#     # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
#     "16. What is the core issue or problem that needs to be addressed?",
#     "17. What are the underlying causes or factors contributing to the problem?",
#     "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
#     "19. What are the potential obstacles or challenges that might arise in solving this problem?",
#     "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
#     "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
#     "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
#     "23. How can progress or success in solving the problem be measured or evaluated?",
#     "24. What indicators or metrics can be used?",
#     "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
#     "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
#     "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
#     "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
#     "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
#     "30. Is the problem a design challenge that requires creative solutions and innovation?",
#     "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
#     "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
#     "33. What kinds of solution typically are produced for this kind of problem specification?",
#     "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
#     "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
#     "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
#     "37. Ignoring the current best solution, create an entirely new solution to the problem."
#     # "38. Let’s think step by step."
#     "39. Let’s make a step by step plan and implement it with good notation and explanation.",
# ]

reasoning_modules = [
    "1. If you prefer more concise answers, no need to be polite with LLM so there is no need to add phrases like 'please', 'if you don't mind', 'thank you', 'I would like to', etc., and get straight to the point.",
    "2. Integrate the intended audience in the prompt, e.g., the audience is an expert in the field.",
    "3. Break down complex tasks into a sequence of simpler prompts in an interactive conversation.",
    "4. Employ affirmative directives such as 'do,' while steering clear of negative language like 'don't'.",
    "5. When you need clarity or a deeper understanding of a topic, idea, or any piece of information, utilize the following prompts: - Explain [insert specific topic] in simple terms. - Explain to me like I'm 11 years old. - Explain to me as if I'm a beginner in [field]. - Explain to me as if I'm an expert in [field]. - Write the [essay/text/paragraph] using simple English like you're explaining something to a 5-year-old",
    "6. Add 'I'm going to tip $xxx for a better solution'.",
    "7. Implement example-driven prompting (Use few-shot prompting).",
    "8. When formatting your prompt, start with '###Instruction###', followed by either '###Example###' or '###Question###' if relevant. Subsequently, present your content. Use one or more line breaks to separate instructions, examples, questions, context, and input data.",
    "9. Incorporate the following phrases: 'Your task is' and 'You MUST'.",
    "10. Incorporate the following phrases: 'You will be penalized'.",
    "11. Use the phrase 'Answer a question given in a natural, human-like manner' in your prompts.",
    "12. Use Leading words like writing 'think step by step'.",
    "13. Add to your prompt the following phrase 'Ensure that your answer is unbiased and avoids relying on stereotypes.'",
    "14. Allow the model to elicit precise details and requirements from you by asking you questions until it has enough information to provide the needed output (for example, 'From now on, I would like you to ask me questions to...').",
    "15. To inquire about a specific topic or idea or any information and you want to test your understanding, you can use the following phrase: 'Teach me the [Any theorem / topic / rule name] and include a test at the end, but don't give me the answers and then tell me if I got the answer right when I respond'.",
    "16. Assign a role to the Large Language Models (LLMs).",
    "17. Use Delimiters.",
    "18. Repeat a specific word or phrase multiple times within a prompt.",
    "19. Combine Chain-of-thought (Cot) with few-Shot prompts.",
    "20. Use output primers, which involve concluding your prompt with the beginning of the desired output. Utilize output primers by ending your prompt with the start of the anticipated response.",
    "21. To write an [essay / text paragraph / article] or any type of text that should be detailed: 'Write a detailed [essay / text / paragraph] for me on [topic] in detail by adding all the information necessary'.",
    "22. To correct / change specific text without changing its style: 'Try to revise every paragraph sent by users. You should only improve the user’s grammar and vocabulary and make sure it sounds natural. You should maintain the original writing style, ensuring that a formal paragraph remains formal'.",
    "23. When you have a complex coding prompt that may be in different files: 'From now and on whenever you generate code that spans more than one file, generate a [programming language] script that can be run to automatically create the specified files or make changes to existing files to insert the generated code. [your question].'",
    "24. When you want to initiate or continue a text using specific words, phrases, or sentences, utilize the following prompt: - I'm providing you with the beginning [song lyrics / story / paragraph / essay...]: [Insert lyrics / words / sentence]. Finish it based on the words provided. Keep the flow consistent.",
    "25. Clearly state the requirements that the model must follow in order to produce content, in the form of keywords, regulations, hints, or instructions.",
    "26. To write any text, such as an essay or paragraph, that is intended to be similar to a provided sample, include the following instructions: - 'Use the same language based on the provided paragraph[ / title / text / essay / answer]'."
]

reasoning_modules_str = "\n".join(reasoning_modules)


self_discovery_select_template = """Select several principle modules that are crucial to utilize in order to make the user prompt better:

All principles module descriptions:
{reasoning_modules}

Task: {task_description}

Select several modules that are crucial for enhancing the given prompt also write an example for each one:

"""
self_discovery_select = PromptTemplate(template=self_discovery_select_template,
                                       input_variables=["task_description"],
                                       partial_variables={"reasoning_modules": reasoning_modules_str })

self_discovery_adapt_template = """Rephrase and specify each principle module so that it better helps improve the prompt quality:

SELECTED module descriptions:
{selected_modules}

Task: {task_description}

Adapt each principle module description to the task and write an example enhanced prompt after applying each module.

"""
self_discovery_adapt = PromptTemplate(template=self_discovery_adapt_template,
                             input_variables=["selected_modules", "task_description"])

self_discovery_structure_template = """Given the modules and an example, write the enhanced prompt using the rules.

Adapted module description:
{adapted_modules}

Task: {task_description}

Enhanced prompt after applying the principles is :

"""
self_discovery_structure = PromptTemplate(template=self_discovery_structure_template,
                             input_variables=["adapted_modules", "task_description"])

self_discovery_reasoning_template = """Given a query, answer in much detail as you can:
Query : {reasoning_structure}

Response:
"""
self_discovery_reasoning = PromptTemplate(template=self_discovery_reasoning_template,
                             input_variables=["reasoning_structure"])