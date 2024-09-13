from detector import JointIntentSlotDetector

model = JointIntentSlotDetector.from_pretrained(
    model_path='./saved_model/model/model_epoch9',
    tokenizer_path='./saved_model/tokenizer/',
    intent_label_path='data/SMP2019/intent_labels.txt',
    slot_label_path='data/SMP2019/slot_labels.txt'
)

print(model.detect('西兰花和猪肉的做法是什么'))