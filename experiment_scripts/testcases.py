import pandas as pd
from transformers import (
    DataCollatorForSeq2Seq,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
import torch
from tqdm import tqdm
from dotenv import dotenv_values
import os
import sys
import json



parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(parent_dir)

from specify_evaluation import JsonBasedEvaluation
from seq_generator_models.python_model.python_seq_generator import pythonSequenceGenerator

config = dotenv_values("../.env")
finetunedmodelpath = os.path.join(parent_dir, config['modelDataFinetuned'].lstrip(os.sep))

casesList = [
    {"code":"""
def my_method(self,):
    str1, str2 = "hello", "general"
    i = 10
    i += 5
    i = newType()
"""},
    {"code":"""
def my_method(self,):
    x: int = 5
    y: str
    z: List[int] = [1, 2, 3]
"""},
    {"code":"""
def my_method():
    for i in range(5):
        print(i.test())
"""},
    {"code":"""
def my_method():
    for key, value in {'a':"1", 'b': 2}.items():
        value.test()
"""},
    {"code":"""
def my_method():
    for x in [1, 2, 3]:
        if check(x):
            break
        print("test")
    else:
        elsecacse()
"""},
    {"code":"""
def my_method():
    async for item in async_iterable:
        asynccase(item)
    else:
        asyncelsecase()
"""},
    {"code":"""
def my_method():
    while x < 5:
        print(x)
        x += 1
"""},
    {"code":"""
def my_method():
    while y > 0:
        if y == 2:
            ifinwhile()
        y -= 1
    else:
        return 0
"""},
    {"code":"""
def my_method():
    if check(x):
        print("x")
        if x < 10:
            print(x)
        else:
            return
    elif check2(x):
        return
"""},
    {"code":"""
def my_method():
    if check(x):
        print("x is greater than 5")
        if x < 10:
            print(x)
        else:
            return
    else:
        return
"""},
    {"code":"""
def my_method():
    with open('file1.txt', 'r') as file1, open('file2.txt', 'w'):
        content = file1.read()
        file2.write(content)
"""},
    {"code":"""
def my_method():
    async with open('file1.txt', 'r') as file1, async_open('file2.txt', 'w') as file2:
        content = await file1.read()
        await file2.write(content)
"""},
    {"code":"""
def my_method():
    match value:
        case 1:
            print("1")
        case True:
            print("true")
        case []:
            print("[]")
        case [1, 2, 3]:
            print("seq")
"""},
    {"code":"""
def my_method():
    match value:
        case {"name": "Alice", "age": 30}:
            print("{'key': 'value'}")
        case {"name": "Bob", **rest}:
            print(f"{rest}")
        case {}:
            print("empty")
"""},
    {"code":"""
def my_method():
    match value:
        case z.Point(1, 2):
            print("point(1,2)")
        case Point(x, y):
            print(f"x={x} and y={y}")
        case Point(1, y=y_value, x=2):
            print(f"x=1 and y={y_value}")
"""},
    {"code":"""
def my_method():
    match value:
        case as_value:
            print(f"{as_value}")
        case 1 | 2 | 3:
            print("1, 2, or 3")
"""},
    {"code":"""
def my_method():
    try:
        x = 1 / 0
    except ZeroDivisionError:
        print("Division by zero")
    else:
        print("No errors")
    finally:
        print("Cleanup")
"""},
    {"code":"""
def my_method():
    try:
        raise
        raise ValueError("error occurred")
        raise RuntimeError("New error occurred") from e
    except:
        pass
"""},
    {"code":"""
def my_method():
    try:
        x = 1 / 0
        raise RuntimeError("error occurred") from e
    except* ZeroDivisionError:
        print("Division by zero")
    except* ValueError:
        print("Value error")
"""},
    {"code":"""
def my_method():
    try:
        x = 1 / 0
    except* ZeroDivisionError:
        print("Division by zero")
    else:
        print("No errors")
    finally:
        print("Cleanup")
"""},
    {"code":"""
def my_method():
    assert x > 0, "x must be positive"
    assert x > 0
    global x
    nonlocal x
"""},
    {"code":"""
def my_method():
    z = x + (y + 5)
    z.get(self.g).test(c, s1=3, s2=hello())
"""},
    {"code":"""
def my_method():
    self.g = z
    z = -z
    c = a if x > 3 else b
"""},
    {"code":"""
def my_method():
    if a == (b+5) and b < 3 and a > 1 or a < 5:
        pass
    def my_method_2(s1, s2):
        s3 = s1 + s2 *3
        return s3
    my_method_2(self.g, c, s1=3, s2=z)
"""},
    {"code":"""
def my_method():
    my_list = [1, 2, 3, 4]
    del my_list[2]
    del my_list[1:3]
    del z
    x = lambda a, b, c : a + b + c
    x = {'a':1, 'b':2}
    x = {'a', 'b', 'c'}
"""},
    {"code":"""
def my_method():
    y = [fun(x) for x in xrange(30) if x<10 and x > 1 if x in my_list]
"""},
    {"code":"""
def my_method():
    y = {x for x in my_list if x % 2 == 0} 
"""},
    {"code":"""
def my_method():
    y = {x:test(x) for x in my_list if x % 2 == 0}
"""},
    {"code":"""
def my_method():
    await task1()
    yield "Hello world!!"
    yield from normal_number_generator
"""},
    {"code":"""
def my_method():
    text = f"Hello, {name}! {25}"
    A=os.environ["USER"][get(10)]
"""},
    {"code":"""
def my_method():
    first, *rest = my_list
    my_list[0:4:1]
    print("Hello, World!")
    print(n:=10)
"""},
    {"code":"""
def my_method():
    sum(x for x in range(10) if x > 5)
    pass
    return self.g + 10
"""},
    {"code":"""
def count_up_to(n):
    count = 1
    while count <= n:
        yield count
        count += 1
"""},
    {"code":"""
def count_up_to(n):
    for number in count_up_to(5):
        print(number)
"""},
    {"code":"""
def fastPathOrderedEmit(value, delayError, disposable):
    observer = downstream;
    q = queue;
    if wip.get() == 0 and wip.compareAndSet(0, 1):
        if q.isEmpty():
            accept(observer, value).get().print()
            if leave(-1) == 0:
                return
        else:
            q.offer(value)
    else:
        q.offer(value)
        if not enter():
            return
    QueueDrainHelper.drainLoop(q, observer, delayError, disposable, this);
"""},
    {"code":"""
def blockingForEach( onNext):
    it = blockingIterable().iterator()
    while it.hasNext():
        try:
            onNext.accept(it.next())
        except Throwable as e:
            Exceptions.throwIfFatal(e)
            it.dispose()
            raise ExceptionHelper.wrapOrThrow(e)
"""},
    {"code":"""
def blockingFirst():
    observer = BlockingFirstObserver()
    subscribe(observer);
    v = observer.blockingGet()
    if v is not null:
        return v
    raise NoSuchElementException()
"""},
    {"code":"""
def toFlowable(strategy):
    f = FlowableFromObservable(self)
    match (strategy):
        case DROP:
            return f.onBackpressureDrop()
        case LATEST:
            return f.onBackpressureLatest()
        case MISSING:
            return f
        case ERROR:
            return RxJavaPlugins.onAssembly(FlowableOnBackpressureError(f))
        case _:
            return f.onBackpressureBuffer()
"""},
    {"code":"""
def to(converter):
    try:
        return ObjectHelper.requireNonNull(converter, "converter is null").apply(this)
    except Throwable as ex:
        Exceptions.throwIfFatal(ex)
        raise ExceptionHelper.wrapOrThrow(ex)
"""},
    {"code":"""
def test():
    match value:
        case 1:
            print("Matched 1")
        case True:
            print("Matched None")
    y = {x for x in my_list.set() if x % 2 == 0} 
    y = {x:test(x) for x in my_list if x % 2 == 0}
"""}
]


def main():
    python_seq_generator = pythonSequenceGenerator()
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-small")
    model = T5ForConditionalGeneration.from_pretrained(finetunedmodelpath)
    device = 'cuda:0'
    model_gpu = model.to(device)
    model_gpu.eval()

    seqs = [json.dumps(python_seq_generator.code_to_seq(code=code['code'])) for code in casesList]
    model_inputs = []
    for element in casesList:
        tokenoutput = tokenizer(element['code'])
        model_inputs.append({
            "input_ids": tokenoutput["input_ids"],
            "attention_mask": tokenoutput["attention_mask"]
        })
    gen_outputs = []
    for sample in tqdm(model_inputs, total=len(model_inputs)):
        source_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model_gpu.generate(
                source_ids,
                do_sample=True,
                num_beams=5,
                max_length=512,
                temperature=0.3,
                top_k=50,
                top_p=0.8,
            )
            gen_outputs.extend(outputs.cpu().tolist())
    gen_dec = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False,) for output in gen_outputs]
    evaluator = JsonBasedEvaluation()
    dataset = []
    for i in range(len(casesList)):
        entry = {
            "code": casesList[i]["code"],
            "input_ids": model_inputs[i]["input_ids"],
            "attention_mask": model_inputs[i]["attention_mask"],
            "seqs": seqs[i],
            "gen_dec": gen_dec[i],
            'evaluation':evaluator.compare_dicts(seqs[i], gen_dec[i])
        }
        dataset.append(entry)
        # print(entry['gen_dec'])

    df = pd.DataFrame(dataset)
    df.to_csv('../dataset/testcase.csv', index=False)

if __name__ == '__main__':
    main()

