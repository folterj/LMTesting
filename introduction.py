from transformers import pipeline


classifier_sentiment = pipeline("sentiment-analysis")
output_sentiment = classifier_sentiment(["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"])


classifier_zs = pipeline("zero-shot-classification")
output_zs = classifier_zs(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)


generator_gen = pipeline("text-generation")
output_gen = generator_gen("In this course, we will teach you how to",
                           max_length=100,
                           num_return_sequences=2,
                           )


unmasker = pipeline("fill-mask")
output_unmask = unmasker("This course will teach you all about <mask> models.", top_k=2)


ner = pipeline("ner", grouped_entities=True)
output_ner = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")


question_answerer = pipeline("question-answering")
output_qa = question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )


summarizer = pipeline("summarization")
output_sum = summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
output_trans = translator("Ce cours est produit par Hugging Face.")
