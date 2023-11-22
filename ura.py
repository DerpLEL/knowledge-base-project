from typing import Any, Dict, List, Mapping, Optional

import requests

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.pydantic_v1 import Extra


class ContentHandlerURAAPIGateway:
    """Adapter to prepare the inputs from Langchain to a format
    that LLM model expects.

    It also provides helper function to extract
    the generated text from the model response."""

    @classmethod
    def transform_input(
        cls, prompt: str, model_kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        res = {"prompt": prompt}
        if model_kwargs.get('lang'):
            res["lang"] = model_kwargs['lang']
        if model_kwargs.get('temprature'):
            res["temprature"] = model_kwargs['temprature']
        return res

    @classmethod
    def transform_output(cls, response: Any) -> str:
        return response.json()['answer']


class URAAPIGateway(LLM):
    """URA API Gateway to access LLM models hosted on DSCLab."""

    api_url: str
    """API Gateway URL"""

    headers: Optional[Dict] = None
    """API Gateway HTTP Headers to send, e.g. for authentication"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    content_handler: ContentHandlerURAAPIGateway = ContentHandlerURAAPIGateway()
    """The content handler class that provides an input and
    output transform functions to handle formats between LLM
    and the endpoint.
    """

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"api_url": self.api_url, "headers": self.headers},
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "URA_api_gateway"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call out to URA API Gateway model.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python

                response = se("Tell me a joke.")
        """
        import json
        _model_kwargs = self.model_kwargs or {}
        payload = self.content_handler.transform_input(prompt, _model_kwargs)
        text = json.dumps(payload)
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                # data=payload,
                timeout=100
            )
            response.raise_for_status()
            text = self.content_handler.transform_output(response)

        # Handle ConnectionError
        except requests.exceptions.ConnectionError as ce:
            print('Connection error:', ce)
        # Handle Timeout
        except requests.exceptions.Timeout as te:
            print('Request timed out:', te)
        # Handle HTTPError
        except requests.exceptions.HTTPError as he:
            print('HTTP error occurred:', he)
        # Handle ValueError
        except ValueError as ve:
            print('JSON decoding error:', ve)
        # Handle Any UnexpectedError
        except Exception as error:
            raise ValueError(f"Error raised by the service: {error}")
        
        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        return text
    
if __name__ == "__main__":
    

    ura_llm = URAAPIGateway(
        headers = {"Content-Type": "application/json; charset=utf-8"},
        api_url = 'https://bahnar.dscilab.com:20007/llama/api',
        model_kwargs={"lang": "vi", "temprature": 0},
    )
    
    prompt = """<s>[INST]
You help extract request_category; student_name; student_id; request_content from the Text. Don't explain anything. If you can't find anything, output empty dictionary.

Text: '&lt i&gt Tốt nghiệp &lt /i&gt . . . .  &lt b&gt Họ tên sinh viên &lt /b&gt . . Nguyễn Quang Trường. . . .  &lt b&gt Mã sinh viên &lt /b&gt . . ABC123. . . .  &lt b&gt Nội dung yêu cầu &lt /b&gt . . Thưa thầy cô,. . em muốn biết em có đủ điều kiện tốt nghiệp không?. . . .'
Assistant: ```json
{"request_category": "Tốt nghiệp",
"student_name": "Nguyễn Quang Trường",
"student_id": "ABC123",
"request_content": "Thưa thầy cô, em muốn biết em có đủ điều kiện tốt nghiệp không?"}```
Text: '< i> Điểm thi < /i> . . . .  < b> Họ tên sinh viên < /b> . . Nguyễn Văn A. . . .  < b> Mã sinh viên < /b> . . BK1126. . . .  < b> Nội dung yêu cầu < /b> . . Chào thầy cô,. . . . Em muốn xin chấm phúc khảo lại bài thi "Tư tưởng Hồ Chí Minh" ạ!. . . . Em cảm ơn.. . . . '
Assistant: ```json
{"request_category": "Điểm thi",
"student_name": "Nguyễn Văn A",
"student_id": "BK1126",
"request_content": "Chào thầy cô, Em muốn xin chấm phúc khảo lại bài thi "Tư tưởng Hồ Chí Minh" ạ! Em cảm ơn."}```
Text: '&lt i&gt Yêu cầu khác &lt /i&gt . . . .  &lt b&gt Họ tên sinh viên &lt /b&gt . . Nguyễn Văn Tám. . . .  &lt b&gt Mã sinh viên &lt /b&gt . . XYZ123. . . .  &lt b&gt Nội dung yêu cầu &lt /b&gt . . Yêu cầu về việc chứng nhận hoàn thành môn học. . . . '
Assistant:  [/INST]"""

    prompt1 = """<s>[INST]

You help extract teacher name; student name from the Text. Don't explain anything.

Text:  < i> Yêu cầu khác < /i> . . . .  < b> Họ tên sinh viên < /b> . . Nguyễn Văn Tám. . . .  < b> Mã sinh viên < /b> . . XYZ123. . . .  < b> Nội dung yêu cầu < /b> . . Yêu cầu về việc chứng nhận hoàn thành môn học. . . .
Assistant: Nguyễn Văn Tám is student name.
Text:  < b> MSSV/MSGV: 1512213 < /b>  . . . .  < i> Điểm thi >  Điểm môn học < /i> . . Vừa qua thầy Nguyễn Thanh Tuấn đã đăng điểm tổng kết môn Xử lí số tín hiệu lên trang điểm của trường , và em có một vài thắc mắc muốn hỏi thầy về điểm thành phần và điểm thi nhưng em không thể tìm thấy thông tin liên lạc của thầy .Mong thầy cô có thể cho em xin thông tin liên lạc của thầy Nguyễn Thanh Tuấn dạy bộ môn Xử lí số tín hiệu học kì vừa qua có được không ạ . Em xin cảm ơn thầy cô ạ . .
Assistant: Nguyễn Thanh Tuấn is teacher name.

Text:  <b>Student ID: 21000082</b> . . . . <i>Graduation > Other...</i> . . . . To the Academic Affairs Department. My name is Nguyen Tran Hoang Anh. - I would like to inquire if the TOEIC score required for graduation for the Class of 2010 is still 450. - I am preparing to submit my TOEIC scores for graduation in November, but during the second semester (2016 - 2017), I registered for courses to maintain my status (I had already completed the previous curriculum), but if my score is below 5 (above 0), will it affect my graduation?
Assistant: [/INST]"""
    print('--' * 30)
    print("The prompt to URA is: ", prompt)
    print('--' * 30)
    print(ura_llm(prompt=prompt))
    exit()
    
    from langchain.llms import AmazonAPIGateway

    api_url = "https://<api_gateway_id>.execute-api.<region>.amazonaws.com/LATEST/HF"
    llm = AmazonAPIGateway(api_url=api_url)

    # These are sample parameters for Falcon 40B Instruct Deployed from Amazon SageMaker JumpStart
    parameters = {
        "max_new_tokens": 100,
        "num_return_sequences": 1,
        "top_k": 50,
        "top_p": 0.95,
        "do_sample": False,
        "return_full_text": True,
        "temperature": 0.2,
    }

    prompt = "what day comes after Friday?"
    llm.model_kwargs = parameters

    llm()