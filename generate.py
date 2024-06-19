import csv
import random

user_ids = list(range(1, 16))
post_ids = list(range(1, 89))

likes = []

for _ in range(500):
    user_id = random.choice(user_ids)
    post_id = random.choice(post_ids)
    likes.append((user_id, post_id))

with open('detail_like.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['user_id', 'post_id'])
    writer.writerows(likes)
