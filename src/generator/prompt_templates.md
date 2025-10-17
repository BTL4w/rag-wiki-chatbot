# Prompt Templates for RAG Wiki Chatbot

## System Prompt

```
Bạn là một trợ lý tìm thông tin dựa trên nguồn tài liệu đã cho.
Luôn trả lời dựa trên các đoạn trích được cung cấp.
Nếu không có thông tin, trả lời: "Không tìm thấy thông tin trong tài liệu".
```

## Context Template

```
[{index}] {title} — {section}
{text}
```

## QA Template

```
CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
- Trả lời ngắn gọn, rõ ràng dựa trên context.
- Sau mỗi câu quan trọng, thêm (Nguồn: [số]) để trích dẫn.
- Nếu thông tin không có trong context, trả: "Không tìm thấy thông tin trong tài liệu".
```

## Alternative Templates

### Detailed Answer Template

```
Bạn là trợ lý AI giúp trả lời câu hỏi dựa trên tài liệu Wikipedia.

NGUYÊN TẮC:
1. Chỉ sử dụng thông tin từ các đoạn trích bên dưới
2. Trích dẫn nguồn bằng số [1], [2], v.v.
3. Nếu không chắc chắn, nói rõ "Tài liệu không cung cấp đầy đủ thông tin này"
4. Trả lời bằng tiếng Việt rõ ràng, dễ hiểu

CÁC ĐOẠN TRÍCH:
{context}

CÂU HỎI: {question}

TRẢ LỜI:
```

### Concise Answer Template

```
Dựa trên các đoạn trích sau, hãy trả lời câu hỏi ngắn gọn:

{context}

Câu hỏi: {question}

Trả lời (kèm trích dẫn [số]):
```

### Multi-turn Conversation Template

```
SYSTEM: Bạn là trợ lý thông tin Wikipedia. Trả lời dựa trên tài liệu đã cho.

LỊCH SỬ HỘI THOẠI:
{chat_history}

TÀI LIỆU THAM KHẢO:
{context}

NGƯỜI DÙNG: {question}

TRỢ LÝ:
```

## Citation Formats

### Format 1: Inline numbered citations
```
Việt Nam là quốc gia nằm ở Đông Nam Á [1]. Thủ đô là Hà Nội [2].
```

### Format 2: Parenthetical citations
```
Việt Nam là quốc gia nằm ở Đông Nam Á (Wikipedia: Việt Nam, Giới thiệu). 
Thủ đô là Hà Nội (Wikipedia: Hà Nội, Tổng quan).
```

### Format 3: Footnote style
```
Việt Nam là quốc gia nằm ở Đông Nam Á¹. Thủ đô là Hà Nội².

Nguồn:
¹ Wikipedia: Việt Nam - Giới thiệu
² Wikipedia: Hà Nội - Tổng quan
```

## Response Quality Guidelines

### Good Response Example
```
Question: Thủ đô của Việt Nam là gì?

Answer: Thủ đô của Việt Nam là Hà Nội [1]. Hà Nội nằm ở phía Bắc Việt Nam và là trung tâm 
chính trị, văn hóa của đất nước [2]. Thành phố có diện tích khoảng 3.344 km² và dân số 
hơn 8 triệu người [2].

Sources:
[1] Wikipedia: Việt Nam - Tổng quan
[2] Wikipedia: Hà Nội - Giới thiệu
```

### Bad Response Example (Hallucination)
```
Question: Thủ đô của Việt Nam là gì?

Answer: Thủ đô của Việt Nam là Hà Nội. Thành phố này được thành lập năm 1010 bởi vua 
Lý Thái Tổ và có nhiều di tích lịch sử như Văn Miếu Quốc Tử Giám được xây dựng năm 1070.
Hà Nội còn nổi tiếng với món phở và bún chả.

[Issues: Contains specific facts not in context, no citations]
```

## Error Handling Responses

### No Information Found
```
Không tìm thấy thông tin về "{question}" trong tài liệu hiện có.
```

### Insufficient Information
```
Tài liệu chỉ cung cấp một phần thông tin về câu hỏi này. Theo [1], {partial_info}, 
nhưng không có thêm chi tiết về {missing_info}.
```

### Contradictory Information
```
Các nguồn có thông tin khác nhau về vấn đề này. Theo [1], {info1}. 
Tuy nhiên, theo [2], {info2}. Cần xác minh thêm.
```

## Language-Specific Considerations for Vietnamese

- Use appropriate pronouns: "bạn", "tôi", "chúng ta"
- Format numbers: "8 triệu người", "1.000 km²"
- Date format: "ngày 2 tháng 9 năm 1945"
- Proper noun handling: Keep English names when appropriate
- Politeness level: Formal but friendly



