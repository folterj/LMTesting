from transformers import pipeline


question_answerer = pipeline("question-answering")
context = "My name is Sylvain and I work at Hugging Face in Brooklyn"
questions = ["Where do I work?", "What's my name?", "Sylvain"]

#context = "I have a monkey. My monkey is very small. It is very lovely. It likes to sit on my head. It can jump very quickly. It is also very clever. It learns quickly. My monkey is lovely. My son has a small dog. His dog is white and sweet. My daughter has a black cat. Her cat is small and clever."
#questions = ["What is monkey?", "How is my monkey?", "Can my monkey jump?", "monkey"]

print("Context:", context)
for question in questions:
    print()
    print("Question:", question)
    answers = question_answerer(question=question, context=context, top_k=5)
    for answer in answers:
        print(f"{answer['answer']} ({answer['score']:.3f})")
