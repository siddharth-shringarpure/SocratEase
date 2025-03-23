import pickle as pk

with open ('logical_model.pk', 'rb') as f:
    logical_model = pk.load(f)

text_illogical = "To solve a complex problem, I would first eat a sandwich because thinking requires energy. Then, I would stare at the problem until it solves itself. If that doesn’t work, I’d randomly press buttons on my keyboard, hoping for a breakthrough. Sometimes, solutions come from dreams, so I might take a nap. If I wake up and the problem is still there, I’d consider asking my cat for advice—cats seem to know things. Finally, if all else fails, I’d flip a coin and let fate decide, because logic is overrated anyway."

text_logical = "To solve a complex problem, I would simply try to do everything at once. I wouldn’t bother breaking it down into smaller parts because that would take too long. Instead, I’d just dive in and start solving everything simultaneously, hoping that it would all magically come together. If something goes wrong, I would ignore it and keep going, because stopping would be a waste of time. As for working with others, I don’t think it’s necessary; it’s faster if I just do everything myself. Eventually, the problem will solve itself, and if it doesn’t, I’ll try again next time, but quicker."


pred_illogical = logical_model.predict(text_illogical)
print(pred_illogical)
pred_logical = logical_model.predict(text_logical)
print(pred_logical)