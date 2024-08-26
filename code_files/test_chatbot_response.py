import os
import pandas as pd
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import AzureChatOpenAI
from dotenv import load_dotenv


class AnswerEvaluator:
    PROMPT_TEMPLATE = """
    Considering the following information:

    Question: {question}
    Context: {ground_truth}
    Answer: {answer}
    Judge how well the answer aligns with the factual information provided in the context and question.

    Additionally, consider these points for a higher score:

    Does the answer directly address the question?
    Does the answer include any statements that contradict the context?
    Does the answer contain any information not supported by the context or question?
    Does the answer completely answer the question?

    For Example
        the answer “France is in western Europe.” to the question “Where is France and what is its capital?” would achieve a low answer relevancy because it only answers half of the question

    Here's how the score is determined:
        0: The answer is completely wrong, contradicts the context, or doesn't address the question.
        0.25 - 0.75: The answer partially addresses the question, might contain irrelevant information, or lacks some details from the context.
        1: The answer directly addresses the question, aligns completely with the context, and doesn't include unsupported information.

    Just return the score nothing else only a floating point number only return 1 when you are completely sure about it.
    """

    def __init__(self):
        load_dotenv()
        self.llm = AzureChatOpenAI(
            default_headers={"User-Id": os.getenv("USER_ID")},
            temperature=0.0,
            deployment_name=os.getenv("DEPLOYMENT_NAME"),
        )
        self.prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE, 
            input_variables=["question", "ground_truth", "answer"]
        )
        self.chain = LLMChain(prompt=self.prompt, llm=self.llm, verbose=False)

    def evaluate_answer_relevancy(self, row):
        try:
            response = self.chain.predict(
                question=row["Question"],
                answer=row["Chatbot Answer"],
                ground_truth=str(row["Answer"]),
            )
            score = float(response) if response else None

            if score is None:
                result = "undetermined"
            elif score < 0.5:
                result = "fail"
            elif score < 1.0:
                result = "pass"
            else:
                result = "correct"
            return score, result
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
            return None, "error"


class TestCaseProcessor:
    def __init__(self, input_filepath, output_filepath):
        self.input_filepath = input_filepath
        self.output_filepath = output_filepath
        self.evaluator = AnswerEvaluator()

    def process_test_cases(self):
        df_results = pd.read_excel(self.input_filepath)
        print("Expected time to process test cases: ", len(df_results))
        df_results[["answer_relevancy", "answer_relevancy_result"]] = df_results.apply(
            lambda row: pd.Series(self.evaluator.evaluate_answer_relevancy(row)), axis=1
        )
        df_results.to_excel(self.output_filepath, index=False)
        print("Test cases processed successfully!")


if __name__ == "__main__":
    input_filepath = "Evaluation-Code-Assignment/testing_results/test_cases_v1.xlsx"
    output_filepath = "final_test_cases.xlsx"
    processor = TestCaseProcessor(input_filepath, output_filepath)
    processor.process_test_cases()
