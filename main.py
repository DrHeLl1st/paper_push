import os
import json
import requests
import yaml
from datetime import datetime, timedelta
from openai import OpenAI

# ============ 读取配置 ============
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

KEYWORDS = config["keywords"]
JOURNALS = config["journals"]
DAYS_BACK = config.get("days_back", 14)

FEISHU_WEBHOOK = os.environ.get("FEISHU_WEBHOOK")
ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY")

# ============ 1. 从 OpenAlex 拉取论文 ============
def fetch_papers(source_id, journal_name, days_back):
    """从 OpenAlex API 获取指定期刊最近 N 天的论文"""
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    url = "https://api.openalex.org/works"
    params = {
        "filter": f"primary_location.source.id:{source_id},from_publication_date:{from_date}",
        "sort": "publication_date:desc",
        "per_page": 200,
        "select": "id,doi,title,publication_date,authorships,abstract_inverted_index,primary_location",
    }
    headers = {"User-Agent": "PaperTracker/1.0 (mailto:your-email@example.com)"}
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        papers = []
        for item in data.get("results", []):
            # 重建摘要（OpenAlex 使用倒排索引存储摘要）
            abstract = reconstruct_abstract(item.get("abstract_inverted_index"))
            authors = [a["author"]["display_name"] for a in item.get("authorships", [])[:5]]
            doi = item.get("doi", "").replace("https://doi.org/", "")
            papers.append({
                "title": item.get("title", "No Title"),
                "doi": doi,
                "url": f"https://doi.org/{doi}" if doi else "",
                "date": item.get("publication_date", ""),
                "authors": ", ".join(authors),
                "abstract": abstract[:1500],  # 截断，节省 token
                "journal": journal_name,
            })
        return papers
    except Exception as e:
        print(f"[ERROR] Fetching {journal_name}: {e}")
        return []


def reconstruct_abstract(inverted_index):
    """将 OpenAlex 的倒排索引摘要还原为正常文本"""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


# ============ 2. AI 筛选相关性 ============
def ai_filter(papers, keywords):
    """使用蚂蚁百灵 判断论文是否与研究方向相关"""
    client = OpenAI(
        api_key=ZHIPU_API_KEY,
        base_url="https://api.ant-ling.com/v1/"
    )
    
    relevant_papers = []
    # 每批处理 10 篇，避免单次 prompt 过长
    batch_size = 10
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i+batch_size]
        paper_list = "\n".join([
            f"[{j+1}] Title: {p['title']}\nAbstract: {p['abstract'][:500]}"
            for j, p in enumerate(batch)
        ])
        
        prompt = f"""你是一个学术论文相关性筛选助手。

我的研究方向关键词：{', '.join(keywords)}

以下是最近发表的论文列表，请判断每篇是否与我的研究方向【高度相关】。
仅选择那些主题直接相关或方法/对象高度匹配的论文。不要选择仅边缘提及的。

论文列表：
{paper_list}

请仅输出相关论文的编号，格式为 JSON 数组，如 [1, 3, 5]。
如果没有相关的，输出空数组 []。不要输出任何其他文字。"""

        try:
            response = client.chat.completions.create(
                model="Ling-2.6-1T",
                # model = "Ling-3.0-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            result_text = response.choices[0].message.content.strip()
            # 提取 JSON 数组
            import re
            match = re.search(r'\[.*?\]', result_text, re.DOTALL)
            if match:
                indices = json.loads(match.group())
                for idx in indices:
                    if 1 <= idx <= len(batch):
                        relevant_papers.append(batch[idx - 1])
        except Exception as e:
            print(f"[ERROR] AI filtering batch {i}: {e}")
            # 降级：关键词匹配
            for p in batch:
                text = (p["title"] + " " + p["abstract"]).lower()
                if any(kw.lower() in text for kw in keywords):
                    relevant_papers.append(p)
    
    return relevant_papers


# ============ 3. 推送到飞书 ============
def send_to_feishu(papers):
    """通过飞书 Webhook 发送论文列表"""
    if not papers:
        msg = "📭 本周期内未发现与你研究方向高度相关的 Nature/Science 论文。"
    else:
        lines = [f"📚 本期筛选出 **{len(papers)}** 篇相关论文：\n"]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"**{i}. {p['title']}**\n"
                f"   期刊: {p['journal']} | 日期: {p['date']}\n"
                f"   作者: {p['authors']}\n"
                f"   链接: {p['url']}\n"
            )
        msg = "\n".join(lines)
    
    payload = {
        "msg_type": "text",
        "content": {"text": msg}
    }
    
    try:
        resp = requests.post(FEISHU_WEBHOOK, json=payload, timeout=10)
        print(f"[Feishu] Status: {resp.status_code}, Response: {resp.text}")
    except Exception as e:
        print(f"[ERROR] Feishu send failed: {e}")


# ============ 主流程 ============
def main():
    print(f"=== Paper Tracker Run: {datetime.utcnow().isoformat()} ===")
    print(f"Keywords: {KEYWORDS}")
    print(f"Journals: {[j['name'] for j in JOURNALS]}")
    print(f"Days back: {DAYS_BACK}")
    
    all_papers = []
    for journal in JOURNALS:
        print(f"\nFetching from {journal['name']}...")
        papers = fetch_papers(journal["source_id"], journal["name"], DAYS_BACK)
        print(f"  Got {len(papers)} papers")
        all_papers.extend(papers)
    
    print(f"\nTotal papers fetched: {len(all_papers)}")
    
    if all_papers:
        print("Running AI filtering...")
        relevant = ai_filter(all_papers, KEYWORDS)
        print(f"Relevant papers: {len(relevant)}")
        send_to_feishu(relevant)
    else:
        print("No papers fetched, skipping.")


if __name__ == "__main__":
    main()
