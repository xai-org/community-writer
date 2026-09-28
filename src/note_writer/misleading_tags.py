import json

from data_models.writer_data_models import MisleadingTag, PostWithContext
from note_writer.llm_client import LLMClient


def _get_prompt_for_misleading_why_tags(post_with_context: PostWithContext, note: str):
    user_name = post_with_context.post.username
    post_id = post_with_context.post.post_id
    post_link = f"https://x.com/{user_name}/status/{post_id}"
    return f"""Below will be a post on X, and a proposed community note that \
adds additional context to the potentially misleading post. \
Your task will be to identify which of the following tags apply to the post and note. \
You may choose as many tags as apply, but you must choose at least one. \
You must respond in valid JSON format, with a list of which of the following options apply:
- "factual_error":  # the post contains a factual error
- "manipulated_media":  # the post contains manipulated/fake/out-of-context media
- "outdated_information":  # the post contains outdated information
- "missing_important_context":  # the post is missing important context
- "disputed_claim_as_fact":  # including unverified claims
- "misinterpreted_satire":  # the post is satire that may likely be misinterpreted as fact
- "other":  # the post contains other misleading reasons

Example valid JSON response:
{{
    "misleading_tags": ["factual_error", "outdated_information", "missing_important_context"]
}}

The post and note are as follows:

{post_link}

Proposed community note:
```
{note}
```
"""


async def get_misleading_tags(
    post_with_context: PostWithContext,
    note_text: str,
    llm_client: LLMClient,
    timeout: float,
    retries: int = 3,
    base_delay: float = 0.0,
) -> list[MisleadingTag]:
    misleading_why_tags_prompt = _get_prompt_for_misleading_why_tags(
        post_with_context, note_text
    )
    grok_output = await llm_client.get_grok_response(
        misleading_why_tags_prompt,
        timeout=timeout,
        expect_json=True,
        retries=retries,
        base_delay=base_delay,
        post_id=post_with_context.post.post_id,
    )
    misleading_why_tags_str = grok_output.content
    if misleading_why_tags_str is None:
        raise ValueError("Grok output content unavailable.")
    try:
        misleading_why_tags = json.loads(misleading_why_tags_str)["misleading_tags"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(
            f"Failed to parse misleading tags from LLM response: {e}\n"
            f"Raw response: {misleading_why_tags_str[:500]}"
        ) from e
    return [MisleadingTag(tag) for tag in misleading_why_tags]
