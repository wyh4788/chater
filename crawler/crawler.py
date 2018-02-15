import praw
import numpy as np
from  __builtin__ import any as b_any

NOT_ALLOWED_KEY_WORDS=["http", "r/", "[removed]", "ad spam site"]

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
        submission.comments.replace_more(limit=None)

        for comment in submission.comments:
            finder(comment)

    question_out = np.asanyarray(question)
    answer_out = np.asanyarray(answer)

    with file('question.out', 'a') as outfile:
        np.savetxt(outfile, question_out, fmt='%s')

    with file('answer.out', 'a') as outfile:
        np.savetxt(outfile, answer_out, fmt='%s')

main()
