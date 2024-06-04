import peewee
import datetime
import base64
import xxhash
from peewee import TextField

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.caching import cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.structs import LmPrediction, LmPrompt

_tables_created = False
_text_hash_len = 32
_text_and_sample_hash_len = 48

database = peewee.SqliteDatabase(cache_dir() / 'cache_db.sqlite3')


class BaseModel(peewee.Model):
    class Meta:
        database = database


class CacheLmPromptText(BaseModel):
    text_hash = peewee.FixedCharField(
        primary_key=True, max_length=_text_hash_len)
    text = TextField()

    @classmethod
    def create_from_prompt(cls, prompt: LmPrompt):
        return cls.get_or_create(
            text_hash=prompt_to_text_hash(prompt),
            text=prompt.get_text_as_string_default_form()
        )[0]


def prompt_to_text_hash(prompt: LmPrompt) -> str:
    text = prompt.get_text_as_string_default_form()
    hasher = xxhash.xxh64()
    hasher.update(text.encode())
    text_hash = base64.b64encode(hasher.digest()).decode()
    # Add the start and end of the text to make it the right length
    #  this ideally also improves locality of the hash
    remaining_chars = _text_hash_len - len(text_hash)
    start_chars = text[:min(remaining_chars // 3, len(text))]
    end_chars = text[-min(remaining_chars - len(start_chars), len(text)):]
    text_hash = start_chars + end_chars[::-1] + text_hash
    # Ensure it is the right length
    if len(text_hash) < _text_hash_len:
        text_hash += "_" * (_text_hash_len - len(text_hash))
    return text_hash


def prompt_to_sample_hash_text(
    prompt: LmPrompt,
    model_key: str,
) -> str:
    return (
        prompt_to_text_hash(prompt)
        + prompt_to_sample_params_hash(prompt, model_key)
    )


def prompt_to_sample_params_hash(
    prompt: LmPrompt,
    model_key: str,
) -> str:
    _target_len = _text_and_sample_hash_len - _text_hash_len
    hasher = xxhash.xxh64()
    hasher.update(
        str(CacheLmPromptSampleParams.prompt_to_only_sample_class_dict(prompt, model_key))
    )
    hash = base64.b64encode(hasher.digest()).decode()
    if len(hash) < _target_len:
        hash += "_" * (_target_len - len(hash))
    elif len(hash) > _target_len:
        hash = hash[:_target_len]
    return hash


class CacheLmPromptSampleParams(BaseModel):
    text_hash = peewee.ForeignKeyField(CacheLmPromptText)
    sample_hash = peewee.FixedCharField(
        max_length=_text_and_sample_hash_len,
        primary_key=True,
    )
    model_key = peewee.CharField()
    max_tokens = peewee.IntegerField(null=True)
    temperature = peewee.FloatField()
    top_p = peewee.FloatField()
    presence_penalty = peewee.FloatField()
    frequency_penalty = peewee.FloatField()
    add_bos_token = peewee.CharField()
    echo = peewee.BooleanField()
    add_special_tokens = peewee.BooleanField()
    has_internals_request = peewee.BooleanField()
    stop = peewee.TextField()

    @classmethod
    def prompt_to_only_sample_class_dict(
        cls,
        prompt: LmPrompt,
        model_key: model_key,
    ):
        return dict(
            model_key=model_key,
            max_tokens=prompt.max_tokens,
            temperature=prompt.temperature,
            top_p=prompt.top_p,
            presence_penalty=prompt.presence_penalty,
            frequency_penalty=prompt.frequency_penalty,
            add_bos_token=str(prompt.add_bos_token),
            echo=prompt.echo,
            add_special_tokens=prompt.add_special_tokens,
            has_internals_request=prompt.model_internals_request is not None,
            stop=str(prompt.stop),
        )

    @classmethod
    def create_from_prompt(
        cls,
        prompt: LmPrompt,
        model_key: str,
    ):
        # Get or create
        textcache = CacheLmPromptText.create_from_prompt(prompt)
        instance, created = cls.get_or_create(
            text_hash=textcache.text_hash,
            sample_hash=prompt_to_sample_hash_text(prompt, model_key),
            **cls.prompt_to_only_sample_class_dict(prompt, model_key),
        )
        return instance


class CacheLmPrediction(BaseModel):
    sample_params = peewee.FixedCharField(
        max_length=_text_and_sample_hash_len,
    )
    base_class = peewee.CharField()
    completion_text = peewee.TextField()
    metad_bytes = peewee.BlobField()
    date_added = peewee.DateTimeField()


def create_tables():
    with database:
        database.create_tables([
            CacheLmPromptText,
            CacheLmPromptSampleParams, CacheLmPrediction
        ])


def _prep_cache():
    global _tables_created
    if not _tables_created:
        create_tables()
        _tables_created = True


def add_prediction_to_cache(
    prediction: LmPrediction,
    model_key: str,
):
    _prep_cache()
    sample_params = CacheLmPromptSampleParams.create_from_prompt(
        prediction.prompt,
        model_key,
    )
    print("sample_params made", sample_params.sample_hash)
    CacheLmPrediction.create(
        sample_params=sample_params,
        base_class=prediction.__class__.__name__,
        completion_text=prediction.completion_text,
        metad_bytes=prediction.serialize_metad_for_cache(),
        date_added=datetime.datetime.now(),
    )


def get_from_cache(
    prompt: LmPrompt,
    lm: LmPredictor = None,
) -> LmPrediction | None:
    _prep_cache()
    sample_hash = prompt_to_sample_hash_text(prompt, lm.model_name())
    res = CacheLmPrediction.select().where(
        CacheLmPrediction.sample_params == sample_hash
    )
    res = list(res)
    assert prompt.num_completions == 1
    if not res:
        return None
    res = res[0]
    return lm.find_prediction_class(prompt).parse_from_cache(
        res.completion_text,
        prompt,
        res.metad_bytes,
    )


def main():
    create_tables()
    lm = get_open_ai_lm()
    pred = lm.predict("Once upon a time")
    add_prediction_to_cache(pred, lm.model_name())


class SqlBackedCache:
    def __init__(self, lm):
        create_tables()
        self._lm = lm

    def __contains__(self, prompt: LmPrompt):
        return get_from_cache(prompt, self._lm) is not None

    def get(self, prompt: LmPrompt):
        return get_from_cache(prompt, self._lm)

    def add(self, prediction: LmPrediction):
        add_prediction_to_cache(prediction, self._lm.model_name())


if __name__ == "__main__":
    main()