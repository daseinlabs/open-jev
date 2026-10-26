"""Generate a sample tickets.csv. Replace it with an export of your own labelled tickets."""
import csv
import random
import sys

TEAMS = {
    'billing': ["I was charged twice for {thing}.", "My invoice for {thing} shows the wrong amount.",
                "Can I get a refund on the {thing} subscription fee?", "Why did my card get billed again this month?"],
    'shipping': ["My {thing} still hasn't arrived.", "The tracking for my {thing} hasn't moved in a week.",
                 "Can you ship the {thing} to a different address?", "The courier left my {thing} at the wrong house."],
    'returns': ["The {thing} arrived broken, I want a replacement.", "How do I send back the {thing}?",
                "The {thing} is the wrong size, can I exchange it?", "I'd like to return the {thing} I bought last week."],
    'account': ["I can't log in to my account.", "Please change the email on my account.",
                "I want to delete my account and data.", "My password reset link isn't working."],
    'technical': ["The app crashes when I open my {thing} order.", "The website shows an error at checkout.",
                  "Your app won't load my order history.", "The {thing} setup guide link is broken."],
}
THINGS = ['headphones', 'blender', 'yoga mat', 'monitor', 'jacket', 'coffee maker', 'backpack', 'desk']
URGENT = ["This is urgent.", "I need this fixed today.", "Please help ASAP!", "This is the third time I'm asking."]
LOW = ["No rush.", "Whenever you get a chance.", "Just curious.", "Not a big deal."]

rng = random.Random(0)
writer = csv.writer(sys.stdout)
writer.writerow(['ticket_id', 'customer_id', 'message', 'team', 'priority'])
for i in range(1200):
    team = rng.choice(list(TEAMS))
    priority = rng.choices(['urgent', 'normal', 'low'], [1, 2, 1])[0]
    message = rng.choice(TEAMS[team]).format(thing=rng.choice(THINGS))
    if rng.random() < .3:  # ambiguous tickets mention a second topic
        other = rng.choice([t for t in TEAMS if t != team])
        message += ' Also, ' + rng.choice(TEAMS[other]).format(thing=rng.choice(THINGS)).lower()
    if rng.random() < .6:
        message += f' Order #{rng.randint(10000, 99999)}.'
    if priority != 'normal':
        message += ' ' + rng.choice(URGENT if priority == 'urgent' else LOW)
    if rng.random() < .05:  # real labels are noisy: some tickets were routed by mistake
        team = rng.choice(list(TEAMS))
    writer.writerow([f't{i:05d}', f'c{rng.randrange(400):04d}', message, team, priority])
