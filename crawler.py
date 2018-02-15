import praw
from  __builtin__ import any as b_any

NOT_ALLOWED_KEY_WORDS=["http", "r/", "[removed]", "ad spam site"]
URL='https://www.reddit.com/r/AntiJokes/comments/7xjgvm/i_finally_got_a_valentine/'

question, answer = [], []

reddit = praw.Reddit(user_agent=USER_AGENT,
                     client_id=CLIENT_ID, client_secret=CLIENT_SECRET,
                     username=USERNAME, password=PASSWORD)

submission = reddit.submission(url=URL)
submission.comments.replace_more(limit=None)

def finder(node):
    replies = node.replies
    if replies:
        if not b_any(keyword in node.body for keyword in NOT_ALLOWED_KEY_WORDS) and not b_any(keyword in replies[0].body for keyword in NOT_ALLOWED_KEY_WORDS):
            question.append(node.body)
            answer.append(replies[0].body)

        for reply in replies:
            finder(reply)

for comment in submission.comments:
    finder(comment)

for q, a in zip(question, answer):
    print q, a
