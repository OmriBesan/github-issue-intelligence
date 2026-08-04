import json

with open('data/processed/scikit-learn_issues_model.jsonl', 'w', encoding='utf-8') as f:
    for i in range(100):
        target = 'Bug' if i < 50 else ('Enhancement' if i < 80 else 'Documentation')
        words = 'crash error bug fix' if i % 2 == 0 else 'feature request documentation add'
        text = f"This is a {target} report number {i} with words {words}"
        record = {
            'issue_id': i,
            'issue_number': i,
            'created_at': f'2023-01-{1 + (i % 28):02}T10:00:00Z',
            'state': 'closed',
            'comments': 0,
            'html_url': '',
            'original_labels': '[]',
            'target': target,
            'raw_title': '',
            'raw_body': '',
            'clean_title': '',
            'clean_body': '',
            'combined_text': text,
            'title_length': 0,
            'body_length': 0,
            'combined_length': len(text)
        }
        f.write(json.dumps(record) + '\n')
