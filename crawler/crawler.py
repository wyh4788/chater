import praw
from  __builtin__ import any as b_any

NOT_ALLOWED_KEY_WORDS=["http", "r/", "[removed]", "ad spam site", "[deleted]"]

question, answer = [], []

def finder(node):
    replies = node.replies

    if replies:
        if not b_any(keyword in node.body for keyword in NOT_ALLOWED_KEY_WORDS) and not b_any(keyword in replies[0].body for keyword in NOT_ALLOWED_KEY_WORDS):
            question.append(node.body.replace('\n', ' '))
            answer.append(replies[0].body.replace('\n', ' '))
        for reply in replies:
            finder(reply)

def main():
    reddit = praw.Reddit(user_agent=USER_AGENT,
                         client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                         username=USERNAME, password=PASSWORD)

    with file('topic_urls.in', 'r') as infile:
        urls = infile.read().splitlines()

    for url in urls:
        submission = reddit.submission(url=url)
        submission.comments.replace_more(limit=3)

        for comment in submission.comments:
            finder(comment)

    with file('question.out', 'a') as outfile:
        for q in question:
            print>>outfile, q.encode('ascii','ignore')

    with file('answer.out', 'a') as outfile:
        for a in answer:
            print>>outfile, a.encode('ascii','ignore')

main()
