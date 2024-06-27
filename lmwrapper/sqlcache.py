import base64
import datetime
import os
import sqlite3
import threading
from dataclasses import dataclass

import xxhash

from lmwrapper.abstract_predictor import LmPredictor
from lmwrapper.caching import cache_dir
from lmwrapper.openai_wrapper import get_open_ai_lm
from lmwrapper.sqlcache_struct import BatchPredictionPlaceholder
from lmwrapper.structs import LmPrediction, LmPrompt

_text_hash_len = 32
_text_and_sample_hash_len = 43

cache_path_fn = lambda: cache_dir() / "lm_cache.db"

# Global variable to hold the database connection
conn = None
conn_lock = threading.Lock()


thread_local = threading.local()


def get_connection():
    if not hasattr(thread_local, "connection") or not cache_path_fn().exists():
        # print("Making new connection")
        thread_local.connection = sqlite3.connect(cache_path_fn(), isolation_level=None)
    else:
        # print("Reusing connection")
        # print("Total changes", thread_local.connection.total_changes)
        pass
    return thread_local.connection


# def get_connection():
#    global conn
#    if conn is None:
#        conn = sqlite3.connect(cache_path_fn(), isolation_level=None)
#    return conn


def close_connection():
    global conn
    if conn is not None:
        conn.close()


def execute_query(
    query: str | list[str | tuple[str, tuple[any, ...]]] | tuple[str, tuple[any, ...]],
    fetchone=False,
    conn=None,
):
    if conn is None:
        conn = get_connection()
    cursor = conn.cursor()
    # with sqlite3.connect(cache_path_fn()) as conn:
    # cursor = conn.cursor()
    if isinstance(query, str):
        cursor.execute(query)
    if isinstance(query, tuple):
        assert len(query) == 2
        cursor.execute(*query)
    if isinstance(query, list):
        for q in query:
            # print("Executing query", q, "with conn", conn)
            # print(f"Database file path: {conn.execute('PRAGMA database_list;').fetchone()}")

            if isinstance(q, str):
                cursor.execute(q)
            elif isinstance(q, tuple):
                assert len(q) == 2
                assert isinstance(q[0], str)
                assert isinstance(q[1], tuple)
                cursor.execute(*q)
            else:
                raise ValueError(f"Unexpected query type {type(q)}")
    conn.commit()
    if fetchone:
        return cursor.fetchone()
    return cursor


def create_tables():
    # TODO: if the file name chances during this process probably need to re-run this
    if cache_path_fn().exists():
        return
    if not os.path.exists(cache_dir()):
        cache_dir().mkdir()

    create_tables_sql = [
        "BEGIN;",
        """
        CREATE TABLE IF NOT EXISTS CacheLmPromptText (
            text_hash TEXT PRIMARY KEY,
            text TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS CacheLmPromptSampleParams (
            text_hash TEXT,
            text_and_sample_hash TEXT PRIMARY KEY,
            model_key TEXT,
            max_tokens INTEGER,
            temperature REAL,
            top_p REAL,
            presence_penalty REAL,
            frequency_penalty REAL,
            add_bos_token TEXT,
            echo INTEGER,
            add_special_tokens INTEGER,
            has_internals_request INTEGER,
            stop TEXT,
            batch_id INTEGER,
            FOREIGN KEY (text_hash) REFERENCES CacheLmPromptText (text_hash),
            FOREIGN KEY (batch_id) REFERENCES Batches (batch_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS CacheLmPrediction (
            text_and_sample_hash TEXT,
            multi_sample_number INTEGER,
            data_populated BOOLEAN,
            base_class TEXT,
            completion_text TEXT,
            metad_bytes BLOB,
            date_added TEXT,
            batch_id INTEGER DEFAULT NULL,
            error_message TEXT DEFAULT NULL,
            UNIQUE (text_and_sample_hash, multi_sample_number) ON CONFLICT REPLACE
        );
        """,
        """
        CREATE INDEX IF NOT EXISTS CacheLmPrediction_text_and_sample_hash_index
        ON CacheLmPrediction (text_and_sample_hash);
        """,
        """
        CREATE TABLE IF NOT EXISTS Batches (
            batch_id TEXT PRIMARY KEY, /* A id we define for this batch */
            user_batch_name TEXT, /* A name given by the user */
            api_id TEXT, /* The id used internally by the api */
            api_category TEXT,  /* eg "openai" */
            status TEXT NOT NULL,
            waiting_for_a_result BOOLEAN NOT NULL DEFAULT 1,
            created_at TEXT,
            total_inputs INTEGER,
            api_json_data TEXT  /* The json-serialized data form the api about the batch */
        );
        """,
    ]
    execute_query(create_tables_sql)


def prompt_to_text_hash(prompt: LmPrompt) -> str:
    text = prompt.get_text_as_string_default_form()
    hasher = xxhash.xxh64()
    hasher.update(text.encode())
    text_hash = base64.b64encode(hasher.digest()).decode()
    remaining_chars = _text_hash_len - len(text_hash)
    # Strip non-alphanumeric characters
    text = "".join(filter(str.isalnum, text))
    start_chars = text[: min(remaining_chars // 3, len(text))]
    end_chars = text[-min(remaining_chars - len(start_chars), len(text)) :]
    text_hash = start_chars + end_chars[::-1] + text_hash
    if len(text_hash) < _text_hash_len:
        text_hash += "_" * (_text_hash_len - len(text_hash))
    return text_hash


def prompt_to_text_and_sample_hash(prompt: LmPrompt, model_key: str) -> str:
    return prompt_to_text_hash(prompt) + prompt_to_sample_params_hash(prompt, model_key)


def prompt_to_sample_params_hash(prompt: LmPrompt, model_key: str) -> str:
    _target_len = _text_and_sample_hash_len - _text_hash_len
    hasher = xxhash.xxh64()
    hasher.update(str(prompt_to_only_sample_class_dict(prompt, model_key)).encode())
    hash = base64.b64encode(hasher.digest()).decode()
    if len(hash) < _target_len:
        hash += "_" * (_target_len - len(hash))
    elif len(hash) > _target_len:
        hash = hash[:_target_len]
    return hash


def prompt_to_only_sample_class_dict(prompt: LmPrompt, model_key: str) -> dict:
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


def create_from_prompt_text(prompt: LmPrompt):
    text_hash = prompt_to_text_hash(prompt)
    text = prompt.get_text_as_string_default_form()
    execute_query(
        "INSERT OR IGNORE INTO CacheLmPromptText (text_hash, text) VALUES (?, ?)",
        (text_hash, text),
    )
    return text_hash


def create_from_prompt_sample_params(prompt: LmPrompt, model_key: str):
    text_hash = create_from_prompt_text(prompt)
    sample_hash = prompt_to_text_and_sample_hash(prompt, model_key)
    params = prompt_to_only_sample_class_dict(prompt, model_key)
    execute_query(
        (
            """
        INSERT OR IGNORE INTO CacheLmPromptSampleParams 
        (text_hash, text_and_sample_hash, model_key, max_tokens, temperature, top_p, presence_penalty, 
        frequency_penalty, add_bos_token, echo, add_special_tokens, has_internals_request, stop) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                text_hash,
                sample_hash,
                params["model_key"],
                params["max_tokens"],
                params["temperature"],
                params["top_p"],
                params["presence_penalty"],
                params["frequency_penalty"],
                params["add_bos_token"],
                params["echo"],
                params["add_special_tokens"],
                params["has_internals_request"],
                params["stop"],
            ),
        ),
    )
    return sample_hash


def add_or_set_prediction_to_cache(prediction: LmPrediction, model_key: str):
    create_tables()
    text_and_sample_hash = prompt_to_text_and_sample_hash(prediction.prompt, model_key)
    params = prompt_to_only_sample_class_dict(prediction.prompt, model_key)
    text_hash = prompt_to_text_hash(prediction.prompt)
    text = prediction.prompt.get_text_as_string_default_form()

    execute_query(
        [
            # "BEGIN;",
            (
                (
                    "INSERT OR IGNORE INTO CacheLmPromptText (text_hash, text) VALUES"
                    " (?, ?);"
                ),
                (text_hash, text),
            ),
            (
                """
            INSERT OR IGNORE INTO CacheLmPromptSampleParams 
            (text_hash, text_and_sample_hash, model_key, max_tokens, temperature, top_p, presence_penalty, 
            frequency_penalty, add_bos_token, echo, add_special_tokens, has_internals_request, stop) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
                (
                    text_hash,
                    text_and_sample_hash,
                    params["model_key"],
                    params["max_tokens"],
                    params["temperature"],
                    params["top_p"],
                    params["presence_penalty"],
                    params["frequency_penalty"],
                    params["add_bos_token"],
                    params["echo"],
                    params["add_special_tokens"],
                    params["has_internals_request"],
                    params["stop"],
                ),
            ),
            (
                """
                INSERT INTO CacheLmPrediction 
                (text_and_sample_hash, multi_sample_number, data_populated, base_class, completion_text, metad_bytes, date_added, error_message)
                VALUES (
                    ?,
                    COALESCE((SELECT MAX(multi_sample_number) FROM CacheLmPrediction WHERE text_and_sample_hash = ?) + 1, 0),
                    ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    text_and_sample_hash,
                    text_and_sample_hash,  # Used twice: once for insertion, once for subquery
                    True,  # data_populated
                    prediction.__class__.__name__,
                    prediction.completion_text,
                    prediction.serialize_metad_for_cache(),
                    datetime.datetime.now().isoformat(),
                    prediction.error_message,
                ),
            ),
        ],
    )


@dataclass
class BatchRow:
    batch_id: str
    user_batch_name: str
    api_id: str
    api_category: str
    status: str
    waiting_for_a_result: bool
    created_at: int
    total_inputs: int
    api_json_data: str


def get_from_cache(
    prompt: LmPrompt,
    lm: LmPredictor = None,
) -> list[LmPrediction | BatchPredictionPlaceholder] | None:
    create_tables()
    sample_hash = prompt_to_text_and_sample_hash(prompt, lm.get_model_cache_key())
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.text_and_sample_hash, p.multi_sample_number, p.data_populated, p.base_class, 
               p.completion_text, p.metad_bytes, p.batch_id, b.api_id, b.api_category, b.status, 
               b.waiting_for_a_result, b.total_inputs, p.error_message
        FROM CacheLmPrediction p
        LEFT JOIN Batches b ON p.batch_id = b.batch_id
        WHERE p.text_and_sample_hash = ?
        ORDER BY p.data_populated DESC, p.multi_sample_number
        LIMIT ?
        """,
        (sample_hash, prompt.num_completions or 1),
    )
    all_ret = cursor.fetchall()
    if not all_ret:
        return None

    out = []
    # If the data is not populated, return a BatchPredictionShell
    for ret in all_ret:
        if not ret[2]:  # data_populated is False
            out.append(
                BatchPredictionPlaceholder(
                    batch_id=ret[6],
                    text_and_sample_hash=ret[0],
                    api_id=ret[7],
                    api_category=ret[8],
                    status=ret[9],
                    waiting_for_a_result=ret[10],
                    batch_total_inputs=ret[11],
                ),
            )
            continue

        # Otherwise, return the LmPrediction object
        completion_text = ret[4]
        metad_bytes = ret[5]
        out.append(
            lm.find_prediction_class(prompt).parse_from_cache(
                completion_text,
                prompt,
                metad_bytes,
                error_message=ret[12],
            ),
        )
    return out


class SqlBackedCache:
    def __init__(self, lm):
        create_tables()
        self._lm = lm

    @property
    def lm(self):
        return self._lm

    def __contains__(self, prompt: LmPrompt):
        return get_from_cache(prompt, self._lm) is not None

    def get(
        self,
        prompt: LmPrompt,
    ) -> list[LmPrediction | BatchPredictionPlaceholder] | None:
        return get_from_cache(prompt, self._lm)

    def add_or_set(self, prediction: LmPrediction):
        add_or_set_prediction_to_cache(prediction, self._lm.get_model_cache_key())

    def update_batch_row(
        self,
        batch_api_id: str,
        status: str,
        waiting_for_a_result: bool,
    ):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Batches 
            SET status = ?, waiting_for_a_result = ?
            WHERE api_id = ?
            """,
            (
                status,
                waiting_for_a_result,
                batch_api_id,
            ),
        )
        conn.commit()

    def put_batch_placeholders(
        self,
        batch_row: BatchRow,
        prompts: list[LmPrompt],
    ):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO Batches (batch_id, user_batch_name, api_id, api_category,"
            " status, waiting_for_a_result, created_at, total_inputs, api_json_data)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                batch_row.batch_id,
                batch_row.user_batch_name,
                batch_row.api_id,
                batch_row.api_category,
                batch_row.status,
                batch_row.waiting_for_a_result,
                batch_row.created_at,
                batch_row.total_inputs,
                batch_row.api_json_data,
            ),
        )
        cursor.executemany(
            """
            INSERT INTO CacheLmPrediction (text_and_sample_hash, data_populated, batch_id)
            VALUES (?, ?, ?)
            """,
            [
                (
                    prompt_to_text_and_sample_hash(
                        prompt,
                        self._lm.get_model_cache_key(),
                    ),
                    False,
                    batch_row.batch_id,
                )
                for prompt in prompts
            ],
        )
        conn.commit()
        return [
            BatchPredictionPlaceholder(
                batch_id=batch_row.batch_id,
                text_and_sample_hash=prompt_to_text_and_sample_hash(
                    prompt,
                    self._lm.get_model_cache_key(),
                ),
                api_id=batch_row.api_id,
                api_category=batch_row.api_category,
                status=batch_row.status,
                waiting_for_a_result=batch_row.waiting_for_a_result,
                batch_total_inputs=batch_row.total_inputs,
            )
            for prompt in prompts
        ]

    def delete(self, prompt: LmPrompt) -> bool:
        """
        Deletes all entries of a prompt (including all the multisamples)
        Returns True if any data was deleted, False otherwise.
        """
        if not isinstance(prompt, LmPrompt):
            raise ValueError(f"Expected LmPrompt, got {type(prompt)}")
        sample_hash = prompt_to_text_and_sample_hash(
            prompt,
            self._lm.get_model_cache_key(),
        )
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM CacheLmPrediction WHERE text_and_sample_hash = ?",
            (sample_hash,),
        )
        cursor.execute(
            "DELETE FROM CacheLmPromptSampleParams WHERE text_and_sample_hash = ?",
            (sample_hash,),
        )
        conn.commit()
        data_deleted = cursor.rowcount > 0
        # Delete the text hash if no longer used
        text_hash = prompt_to_text_hash(prompt)
        cursor.execute(
            "SELECT COUNT(*) FROM CacheLmPromptSampleParams WHERE text_hash = ?",
            (text_hash,),
        )
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "DELETE FROM CacheLmPromptText WHERE text_hash = ?",
                (text_hash,),
            )
            conn.commit()
        return data_deleted


def main():
    create_tables()
    lm = get_open_ai_lm()
    pred = lm.predict("Once upon a time")
    add_or_set_prediction_to_cache(pred, lm.get_model_cache_key())
    # add_prediction_to_cache(pred, lm.get_model_cache_key())


if __name__ == "__main__":
    main()
