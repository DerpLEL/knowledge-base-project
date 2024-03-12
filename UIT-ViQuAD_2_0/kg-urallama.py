import random
import json
from urallama import URAWrapper

# import google.generativeai as genai
#
# GOOGLE_API_KEY='AIzaSyAnT0-DpdDE63wJpH51BT3GiB1n8e_tFNo'
#
# genai.configure(api_key=GOOGLE_API_KEY)
#
# model = genai.GenerativeModel('gemini-pro')

model = URAWrapper()

with open('kg_context_viquad.json', 'r', encoding='utf-8') as f:
    kg_context = json.load(f)

with open('possible-questions-viquad.json', encoding='utf-8') as f:
    possible_questions = json.load(f)

with open('impossible-questions-viquad.json', encoding='utf-8') as f:
    impossible_questions = json.load(f)

random.seed(27)
chosen_possible_set = random.choices(possible_questions, k=60)
random.seed(27)
chosen_impossible_set = random.choices(impossible_questions, k=40)

chosen_set = chosen_possible_set + chosen_impossible_set

# prompt_format = '''[INST] Given context, answer the question. Output "N/A" if no answer can be found.
#
# Context: {context}
#
# Question: {question}
# Answer: Here is the most relevant answer from the context: [/INST]'''

prompt_format = '''[INST] Cho một đoạn ngữ cảnh, hãy trả lời câu hỏi. Trả lời ngắn gọn.
Phản hồi "N/A" nếu không có câu trả lời.

Ngữ cảnh: {context}

Câu hỏi: {question}

Trả lời: [/INST]'''

# result_format = '''Context: {context}
#
# Question: {question}
# Answer: {answer}
#
# Reference answer: {ref_answer}'''

qa_result = []

bugged_questions = []
for index, i in enumerate(chosen_set):
    # if index == 41 or index == 69:
    #     continue
    if i['context'] == 'Văn hóa Nhị Lý Đầu đã tạo ra bước nhảy từ thời đại đồ đá mới sang thời đại đồ đồng tại Trung Quốc, nền văn hóa này được đặt tên theo di chỉ Nhị Lý Đầu ở thôn cùng tên tại Yển Sư của Hà Nam. Nền văn hóa này tồn tại trong khoảng thời gian từ 1880 TCN đến 1520 TCN, Di chỉ Nhị Lý Đầu có dấu tích của các tòa cung điện và các xưởng nấu chảy đồng. Nhị Lý Đầu là nơi duy nhất sản xuất các bình đồng dùng trong lễ nghi đương thời tại Trung Quốc. Thời gian tồn tại của văn hóa Nhị Lý Đầu tương ứng với vương triều Hạ, quốc gia liên minh bộ lạc hay triều đại đầu tiên và có tính thần thoại cao trong lịch sử Trung Quốc. Khu vực Hà Nam ngày nay là trung tâm của vương triều Hạ, khu vực di chỉ Nhị Lý Đầu được nhiều học giả nhận định là đô thành trong toàn bộ thời gian hoặc thời kỳ thứ nhất, thứ hai của vương triều Hạ, song vẫn đang phải tìm kiếm các cơ sở vững chắc để làm rõ.':
        context = '''- Văn hóa Nhị Lý Đầu [tồnTạiTrong] 1880 TCN - 1520 TCN
- Văn hóa Nhị Lý Đầu [đượcĐặtTênTheo] Di chỉ Nhị Lý Đầu
- Di chỉ Nhị Lý Đầu [thuộcVề] Yển Sư
- Yển Sư [thuộcVề] Hà Nam
- Văn hóa Nhị Lý Đầu [sảnXuất] Bình đồng
- Bình đồng [dùngTrong] Lễ nghi
- Văn hóa Nhị Lý Đầu [tươngỨngVới] Vương triều Hạ
- Vương triều Hạ [là] Quốc gia liên minh bộ lạc
- Vương triều Hạ [là] Triều đại đầu tiên
- Vương triều Hạ [cóTínhChất] Thần thoại
- Hà Nam [là] Trung tâm
- Vương triều Hạ [có] Di chỉ Nhị Lý Đầu
- Di chỉ Nhị Lý Đầu [đượcNhậnĐịnhLà] Đô thành
- Đô thành [thuộcVề] Vương triều Hạ'''

    elif i['context'] == 'Đối với thuế thân, ông cũng cho giảm từ 120 tiền xuống còn 40 tiền. Với việc lao dịch, trước đây mỗi năm người dân phải đi 1 lần, ông ban chiếu giảm xuống còn 3 năm 1 lần. Mỗi khi có thiên tai, ông thường ra lệnh cho chư hầu không cần tiến cống, lại xoá lệnh bỏ cấm núi đầm, tức là mở cửa những núi đầm của hoàng gia cho nhân dân có thể qua lại hái lượm, đánh bắt trong đó kiếm ăn qua thời mất mùa. Ngoài ra, ông còn nhiều lần hạ chiếu cấm các châu quận cống hiến những kỳ trân dị vật. Trong giai đoạn đầu, nhà Hán đang ở thời kỳ khôi phục kinh tế; tài chính và vật tư đều thiếu thốn. Trước bối cảnh đó, Hán Văn đế chi dùng rất tiết kiệm. Ông trở thành vị vua tiết kiệm nổi tiếng trong lịch sử Trung Quốc.':
        context = '''- Hán Văn đế [ban hành] chiếu giảm thuế thân từ 120 tiền xuống còn 40 tiền
- Hán Văn đế [ban hành] chiếu giảm lao dịch từ 1 lần/năm xuống còn 3 năm/lần
- Hán Văn đế [ban hành] lệnh xoá bỏ cấm núi đầm
- Hán Văn đế [ban hành] chiếu cấm các châu quận cống hiến kỳ trân dị vật
- Hán Văn đế [chi dùng] tiết kiệm
- Nhà Hán [ở trong giai đoạn] khôi phục kinh tế
- Nhà Hán [thiếu thốn] tài chính và vật tư'''

    else:
        context = kg_context[i['context']]

    question = i['question']

    try:
        answer = model.predict(
            prompt_format.format(
                context=context,
                question=question
            )
        )

        if answer == -1:
            raise Exception('Custom.')

    except Exception:
        print(f'Bugged question {index}, skipping...')
        bugged_questions.append(index)
        continue

    # result_formatted = result_format.format(
    #     context=context,
    #     question=question,
    #     answer=answer,
    #     ref_answer=str(i['answers'])
    # )

    plausible_answer = [] if 'plausible_answers' not in i else i['plausible_answers']

    dct = {
        'id': index,
        'question': question,
        'answer': answer,
        'ref_answer': i['answers'],
        'context': context,
        'plausible_answers': plausible_answer
    }
    qa_result.append(dct)
    print(f'Answered question {index}: {question}')

print(f'### SKIPPED QUESTIONS: {bugged_questions} ###')

with open('kg-urallama-viquad-result.json', 'w', encoding='utf-8') as f:
    json.dump(qa_result, f, ensure_ascii=False)
