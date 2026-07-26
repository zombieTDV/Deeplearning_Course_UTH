## 1\. Xác định phạm vi sử dụng AI

Trước khi làm dự án, nhóm thống nhất:

| Công việc | AI hỗ trợ | Con người chịu trách nhiệm |
| --- | --- | --- |
| Phân tích yêu cầu | Gợi ý chức năng, phân tích nghiệp vụ | Nhóm quyết định yêu cầu cuối |
| Thiết kế hệ thống | Gợi ý kiến trúc, database, flow | Nhóm kiểm tra tính phù hợp |
| Lập trình | Sinh code mẫu, giải thích lỗi | Thành viên review, sửa code |
| Testing | Tạo test case, tìm lỗi | Nhóm chạy test thực tế |
| Viết tài liệu | Hỗ trợ format, diễn đạt | Nhóm kiểm tra nội dung |

Ví dụ:

> AI được sử dụng để hỗ trợ thiết kế database, giải thích lỗi code Python, tạo test case và hỗ trợ viết tài liệu. Nhóm chịu trách nhiệm kiểm tra, chỉnh sửa và xác nhận toàn bộ sản phẩm.

---

## 2\. Tạo hồ sơ sử dụng AI (AI Usage Log)

Mỗi lần dùng AI nên lưu lại thành bảng.

Ví dụ tạo file:

```
docs/
 └── AI_USAGE_LOG.md
```

Nội dung:

```markdown
# AI Usage Log

## Project:
Hệ thống quản lý cứu trợ thiên tai

## AI Tools Used

| Tool | Version/Model | Purpose |
|-|-|-|
| ChatGPT | GPT-5.5 | Analyze requirements, debugging |
| GitHub Copilot | Latest | Code suggestion |
| Gemini | 2.x | Documentation support |

---

# Usage Records

## Record 01

### Date:
2026-07-25

### Purpose:
Thiết kế database

### Original Prompt:

"Thiết kế database cho hệ thống quản lý cứu trợ thiên tai gồm người dùng, khu vực, hàng cứu trợ..."

### Modified Prompt:

"Thiết kế database quan hệ sử dụng MySQL, chuẩn hóa đến 3NF, bao gồm bảng User, Donation, ReliefRequest..."

### AI Response:

AI đề xuất các bảng:
- User
- Donation
- Location
- Request

### Accepted:

Một phần

### Modified:

- Thêm bảng Inventory
- Điều chỉnh quan hệ User - Role
- Bỏ một số thuộc tính không cần thiết

### Reason:

Thiết kế lại để phù hợp với yêu cầu môn học.

### Verification:

- Kiểm tra ERD
- Chạy thử tạo database
- Test CRUD

### Commit:

commit:
"create database schema"
```

---

## 3\. Quy trình khi dùng AI để viết code (Vibe Coding Flow)

Nên thống nhất flow:

```
Requirement
      |
      v
Viết prompt
      |
      v
AI sinh giải pháp/code
      |
      v
Developer review
      |
      v
Chỉnh sửa code
      |
      v
Testing
      |
      v
Commit Git
```

Không commit trực tiếp code AI tạo ra.

Ví dụ:

Sai:

```
Prompt -> Copy code -> Commit
```

Đúng:

```
Prompt
 |
AI tạo code
 |
Đọc hiểu
 |
Sửa
 |
Test
 |
Commit
```

---

## 4\. Lưu prompt gốc và prompt chỉnh sửa

Tạo thư mục:

```
docs/
 └── prompts/
      ├── database_prompt.md
      ├── frontend_prompt.md
      ├── debugging_prompt.md
```

Ví dụ:

```markdown
# Database Design Prompt

## Original Prompt

"Tạo database quản lý sinh viên"

## Improved Prompt

"Thiết kế database MySQL cho hệ thống quản lý sinh viên đại học.
Yêu cầu:
- Chuẩn hóa 3NF
- Có khóa chính, khóa ngoại
- Có ERD
- Có giải thích quan hệ"

## Why Modified?

Prompt ban đầu thiếu:
- Công nghệ sử dụng
- Ràng buộc dữ liệu
- Mục tiêu hệ thống
```

---

## 5\. Lưu kết quả AI trả về

Không cần lưu toàn bộ hội thoại dài.

Chỉ cần:

```
docs/
 └── ai_outputs/
      ├── architecture.md
      ├── database.md
      └── test_cases.md
```

Ví dụ:

```markdown
# AI Output - Database

AI suggested:

Tables:
- Users
- Orders
- Products

Team decision:

Accepted:
- Users
- Products

Rejected:
- Orders

Reason:
Không thuộc phạm vi dự án.
```

---

## 6\. Review và kiểm chứng

Mỗi phần AI tạo ra cần có bảng review:

| Nội dung | AI tạo | Người kiểm tra | Kết quả |
| --- | --- | --- | --- |
| Database | Có | Thành viên Backend | Đã sửa |
| API | Có | Backend leader | OK |
| UI | Có | Frontend leader | Sửa |
| Documentation | Có | Leader | OK |

---

## 7\. Commit Git phải thể hiện phần AI hỗ trợ

Ví dụ commit:

```
feat(database): design user schema

AI assistance:
- Used ChatGPT for initial ERD suggestion

Human modification:
- Added role permission
- Changed relationship structure

Testing:
- Verified SQL migration
```

Hoặc đơn giản:

```
docs: add AI usage report

AI-assisted:
- Requirement analysis
- Documentation formatting
```

---

## 8\. Cấu trúc thư mục nên có trong repo

Ví dụ:

```
Project/

├── src/
│
├── tests/
│
├── docs/
│   ├── SRS.md
│   ├── Design.md
│   ├── AI_USAGE_LOG.md
│   ├── prompts/
│   │     ├── coding.md
│   │     ├── design.md
│   │
│   └── ai_outputs/
│         ├── analysis.md
│
├── README.md
└── .git/
```

---

## 9\. Phân công trong nhóm

Nên có một người quản lý AI Documentation:

| Vai trò | Nhiệm vụ |
| --- | --- |
| Leader | Kiểm tra tuân thủ quy trình |
| Developer | Lưu prompt/code AI liên quan |
| Tester | Lưu kết quả kiểm thử |
| Documenter | Tổng hợp AI Usage Report |

---

## 10\. Khi nộp sản phẩm

Trong báo cáo cuối nên có mục:

## AI Usage Statement

Ví dụ:

> Nhóm sử dụng AI như công cụ hỗ trợ trong quá trình phân tích yêu cầu, thiết kế hệ thống, lập trình và kiểm thử. Các kết quả do AI sinh ra đều được thành viên nhóm kiểm tra, chỉnh sửa và xác nhận trước khi tích hợp vào sản phẩm. Nhóm chịu trách nhiệm hoàn toàn về tính chính xác, bảo mật và khả năng vận hành của hệ thống.

---

Nếu làm đúng, chỉ cần thêm 3 thứ vào repo là gần như đáp ứng toàn bộ yêu cầu trong ảnh:

1. `AI_USAGE_LOG.md` → chứng minh quá trình dùng AI
2. `prompts/` → lưu prompt gốc + prompt cải tiến
3. Commit message có ghi phần AI hỗ trợ + phần nhóm chỉnh sửa

Đây là cách phù hợp nhất với yêu cầu "AI chỉ là công cụ hỗ trợ, sinh viên chịu trách nhiệm cuối cùng".

---

**You said**

CHI TIẾT HOÀN CHỈNH HƠN

---

**ChatGPT said**

Dựa trên yêu cầu trong ảnh, nhóm nên xây dựng một **quy trình quản lý việc sử dụng AI (AI-Assisted Development Workflow)** hoàn chỉnh, có thể đưa trực tiếp vào báo cáo dự án hoặc repository GitHub.

Mục tiêu của quy trình:

- Chứng minh nhóm **được phép sử dụng AI** nhưng không phụ thuộc hoàn toàn vào AI.
- Lưu lại toàn bộ quá trình: **dùng công cụ gì → dùng prompt nào → AI trả kết quả gì → nhóm chỉnh sửa gì → kiểm thử ra sao**.
- Đảm bảo có thể truy xuất lại quá trình phát triển sản phẩm.

---

## QUY TRÌNH SỬ DỤNG AI VÀ VIBE CODING TRONG DỰ ÁN

## 1\. Nguyên tắc chung

Trong quá trình thực hiện dự án, nhóm được phép sử dụng các công cụ AI nhằm hỗ trợ:

- Phân tích yêu cầu hệ thống.
- Nghiên cứu công nghệ.
- Thiết kế kiến trúc phần mềm.
- Thiết kế cơ sở dữ liệu.
- Hỗ trợ lập trình.
- Debug và tìm lỗi.
- Viết tài liệu kỹ thuật.
- Tạo test case.
- Cải thiện chất lượng mã nguồn.

Tuy nhiên:

- AI chỉ đóng vai trò là công cụ hỗ trợ.
- Thành viên nhóm chịu trách nhiệm kiểm tra toàn bộ nội dung do AI tạo ra.
- Không sử dụng trực tiếp kết quả AI mà không qua quá trình đánh giá.
- Không đưa dữ liệu nhạy cảm, mã nguồn bảo mật hoặc thông tin cá nhân vào công cụ AI công cộng.

---

## 2\. Danh sách công cụ AI sử dụng

Tạo file:

```
docs/AI_TOOLS.md
```

Nội dung:

```markdown
# AI Tools Used in Project

## 1. ChatGPT

Model:
GPT-5.5

Purpose:
- Requirement analysis
- System design suggestion
- Debugging assistance
- Documentation support

Usage level:
Medium

## 2. GitHub Copilot

Purpose:
- Code completion
- Generate repetitive code
- Explain existing code

Usage level:
Low

## 3. Gemini

Purpose:
- Research
- Compare technical solutions

Usage level:
Low
```

---

## 3\. Quy trình sử dụng AI tổng thể

Quy trình chuẩn:

```
Problem Definition
        |
        v
Human Analysis
        |
        v
Create Prompt
        |
        v
AI Generate Solution
        |
        v
Human Review
        |
        v
Modify / Improve
        |
        v
Testing & Verification
        |
        v
Commit Source Code
        |
        v
Documentation
```

Giải thích:

## Bước 1: Xác định vấn đề

Con người phải hiểu vấn đề trước khi hỏi AI.

Ví dụ:

Không hỏi:

```
Viết code quản lý sinh viên
```

Mà phải:

```
Tôi đang xây dựng hệ thống quản lý sinh viên.
Backend sử dụng Python Flask.
Database MySQL.
Cần thiết kế API CRUD cho bảng Student.
Yêu cầu:
- REST API
- Validate dữ liệu
- Có xử lý lỗi.
```

---

## 4\. Quy trình tạo Prompt

Tạo thư mục:

```
docs/
└── prompts/
```

Mỗi prompt cần có:

- Mục đích.
- Prompt ban đầu.
- Prompt cải tiến.
- Lý do chỉnh sửa.

Ví dụ:

```
docs/prompts/database_design.md
```

Nội dung:

```markdown
# Database Design Prompt

## Objective

Thiết kế database cho hệ thống quản lý cứu trợ thiên tai.

## Original Prompt

Thiết kế database cho hệ thống cứu trợ.

## Problem

Prompt ban đầu thiếu:
- Công nghệ database
- Quy mô hệ thống
- Quan hệ giữa bảng
- Ràng buộc dữ liệu

## Improved Prompt

Thiết kế database MySQL cho hệ thống quản lý cứu trợ thiên tai.

Yêu cầu:

- Chuẩn hóa dữ liệu 3NF.
- Có Primary Key.
- Có Foreign Key.
- Có ERD.
- Giải thích quan hệ giữa các bảng.

## Reason for Improvement

Prompt chi tiết hơn giúp AI tạo kết quả phù hợp với yêu cầu dự án.
```

---

## 5\. Hồ sơ sử dụng AI (AI Usage Log)

Đây là phần quan trọng nhất.

Tạo:

```
docs/AI_USAGE_LOG.md
```

Mẫu:

```markdown
# AI Usage Log

Project:

Hệ thống quản lý cứu trợ thiên tai

---

# Record 01

## Date

25/07/2026

## Member

Nguyễn Văn A

## AI Tool

ChatGPT GPT-5.5

## Task

Thiết kế database

## Goal

Tạo mô hình database ban đầu.

## Original Prompt

Thiết kế database cho hệ thống quản lý cứu trợ thiên tai.

## Improved Prompt

Thiết kế database MySQL cho hệ thống quản lý cứu trợ thiên tai.

Bao gồm:

- User management
- Disaster information
- Donation
- Inventory
- Relief distribution

Yêu cầu chuẩn hóa 3NF.

## AI Response Summary

AI đề xuất:

Tables:

- Users
- Disaster
- Donation
- Inventory
- Distribution

## Decision

Accepted partially.

## Human Modification

Nhóm chỉnh sửa:

- Thêm bảng Role.
- Thêm bảng Location.
- Thay đổi quan hệ Donation - Inventory.

## Reason

Phù hợp hơn với nghiệp vụ thực tế.

## Verification

Kiểm tra:

✓ ER Diagram

✓ SQL migration

✓ CRUD testing

## Commit

Commit:

feat(database): create initial schema
```

---

## 6\. Quy trình sử dụng AI khi lập trình (Vibe Coding)

## 6.1 Không copy code trực tiếp

Quy trình:

```
AI Code Suggestion

        ↓

Developer đọc hiểu

        ↓

Kiểm tra:

- Logic
- Security
- Performance
- Coding style

        ↓

Chỉnh sửa

        ↓

Test

        ↓

Commit
```

---

Ví dụ:

AI tạo:

```python
def login(username,password):
    query = "SELECT * FROM users WHERE username='"+username+"'"
```

Developer phải nhận ra lỗi:

- SQL Injection.
- Không hash password.

Sau đó sửa thành:

```python
Prepared statement

Password hashing

Input validation
```

Trong AI log ghi:

```markdown
AI generated authentication code.

Rejected because:

- SQL injection vulnerability.
- No password encryption.

Modified:

- Added bcrypt hashing.
- Added parameterized query.
```

---

## 7\. Quy trình Review kết quả AI

Mỗi kết quả AI phải đánh giá theo bảng:

```
AI Review Checklist
```

| Tiêu chí | Kiểm tra |
| --- | --- |
| Correctness | Kết quả có đúng không |
| Security | Có lỗi bảo mật không |
| Performance | Có ảnh hưởng hiệu năng không |
| Maintainability | Có dễ bảo trì không |
| Compatibility | Có phù hợp công nghệ không |

Ví dụ:

| AI Output | Kết quả |
| --- | --- |
| Database schema | Chỉnh sửa |
| API code | Chấp nhận sau review |
| UI layout | Tham khảo |
| Documentation | Chỉnh sửa câu chữ |

---

## 8\. Kiểm thử kết quả AI

Sau khi sử dụng AI:

## Code

Kiểm tra:

- Unit test.
- Integration test.
- Manual test.

Ví dụ:

```markdown
AI generated:

Function:
calculate_total()

Verification:

Input:
10 products

Expected:
500000

Actual:
500000

Result:
PASS
```

---

## 9\. Quy trình Commit Git

Mỗi commit liên quan AI nên ghi rõ.

Format:

```
<type>: <description>

AI assistance:
- What AI helped with

Human changes:
- What developer modified

Testing:
- Verification result
```

Ví dụ:

```
feat(api): create student API

AI assistance:
- Generated initial API structure

Human changes:
- Added authentication middleware
- Fixed validation logic

Testing:
- Tested CRUD operations
```

---

## 10\. Cấu trúc Repository đề xuất

```
Project
│
├── src
│   ├── backend
│   └── frontend
│
├── tests
│
├── docs
│   │
│   ├── SRS.md
│   ├── Architecture.md
│   ├── Database.md
│   │
│   ├── AI_USAGE_LOG.md
│   ├── AI_TOOLS.md
│   │
│   ├── prompts
│   │   ├── analysis.md
│   │   ├── coding.md
│   │   ├── database.md
│   │
│   └── ai_outputs
│       ├── design.md
│       ├── testcases.md
│
├── README.md
│
└── CHANGELOG.md
```

---

## 11\. Phân công trách nhiệm trong nhóm

| Vai trò | Trách nhiệm |
| --- | --- |
| Project Leader | Kiểm tra quy trình AI |
| Developer | Lưu prompt/code AI liên quan |
| Tester | Kiểm chứng kết quả AI |
| Document Manager | Tổng hợp AI report |

---

## 12\. Báo cáo cuối dự án

Trong báo cáo thêm chương:

## AI Usage Report

## 1\. AI Contribution

AI được sử dụng để:

- Hỗ trợ phân tích yêu cầu.
- Đề xuất kiến trúc.
- Hỗ trợ lập trình.
- Tạo tài liệu.

## 2\. Human Responsibility

Nhóm đã:

- Review kết quả AI.
- Sửa lỗi.
- Kiểm thử.
- Quyết định sử dụng hoặc loại bỏ.

## 3\. Example Cases

| Task | AI Support | Human Action |
| --- | --- | --- |
| Database | Suggest schema | Modify ERD |
| Coding | Generate template | Rewrite logic |
| Testing | Generate test case | Execute test |

---

## 13\. Checklist trước khi nộp

## AI Documentation

☑ Có danh sách AI tools  
☑ Có version/model  
☑ Có mục đích sử dụng  
☑ Có prompt gốc  
☑ Có prompt cải tiến  
☑ Có kết quả AI  
☑ Có phần chỉnh sửa  
☑ Có lý do chỉnh sửa  
☑ Có kiểm thử  
☑ Có commit liên quan

## Source Code

☑ Không commit code AI chưa review  
☑ Có lịch sử Git rõ ràng  
☑ Có test kết quả  
☑ Thành viên hiểu được toàn bộ code

---

Quy trình này đáp ứng đầy đủ 9 yêu cầu trong ảnh:

1. Công cụ + phiên bản AI → `AI_TOOLS.md`
2. Ngày/mục tiêu sử dụng → `AI_USAGE_LOG.md`
3. Prompt gốc + chỉnh sửa → `prompts/`
4. Tệp hoặc thành phần AI hỗ trợ → log từng record
5. Kết quả AI → `ai_outputs/`
6. Phần chấp nhận/chỉnh sửa/bỏ → Decision + Modification
7. Lý do chỉnh sửa → Reason
8. Kiểm thử/xác minh → Verification
9. Commit tương ứng → Git history

Đây là dạng quy trình có thể dùng cho đồ án lớn, môn lập trình, AI engineering hoặc software engineering.