import base64
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlencode

from requests_oauthlib import OAuth1Session  # type: ignore

from data_models.writer_data_models import (
    Media,
    Post,
    PostWithContext,
    PublicMetrics,
    SuggestedSource,
    UserPublicMetrics,
)


@dataclass
class FeedFetchResult:
    posts: list[PostWithContext]
    pages_fetched: int
    error_count: int
    elapsed_seconds: float
    oldest_timestamp_ms: int


class PostRetrievalError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Post retrieval failed (status {status_code}): {message}")


def _fetch_posts_eligible_for_notes(
    oauth: OAuth1Session,
    post_selection: str,
    max_results: int = 25,
    pagination_token: str | None = None,
) -> dict:
    base_url = "https://api.x.com/2/notes/search/posts_eligible_for_notes"
    post_selection_value = f"feed_size: {post_selection}, feed_lang: all"
    params = {
        "test_mode": "false",
        "max_results": max_results,
        "post_selection": post_selection_value,
        "tweet.fields": "author_id,created_at,referenced_tweets,media_metadata,note_tweet,suggested_source_links_with_counts,note_request_suggestions,lang,public_metrics",
        "expansions": "author_id,attachments.media_keys,referenced_tweets.id",
        "user.fields": "username,verified_type,public_metrics,parody",
        "media.fields": "alt_text,duration_ms,height,media_key,preview_image_url,public_metrics,type,url,width,variants",
    }
    if pagination_token:
        params["pagination_token"] = pagination_token
    url = f"{base_url}?{urlencode(params)}"
    response = oauth.get(url)
    if not response.ok:
        raise PostRetrievalError(response.status_code, response.text)
    return response.json()


def _parse_individual_post(
    item: dict, media_by_key: dict[str, dict], users_by_id: dict[str, dict]
) -> Post:
    media_objs: list[Media] = []
    media_keys = item.get("attachments", {}).get("media_keys", [])

    for key in media_keys:
        if key in media_by_key:
            media_obj = Media(**media_by_key[key])
            media_objs.append(media_obj)

    text = item["text"]
    note_tweet_text = item.get("note_tweet", {}).get("text", "")
    if note_tweet_text:
        text = note_tweet_text

    author_id = item["author_id"]
    user_data = users_by_id.get(author_id, {})
    username = user_data.get("username", "unknown")
    author_verified_type = user_data.get("verified_type")
    author_parody = user_data.get("parody")

    raw_author_metrics = user_data.get("public_metrics")
    author_public_metrics = (
        UserPublicMetrics(**raw_author_metrics) if raw_author_metrics else None
    )

    lang = item.get("lang")

    raw_metrics = item.get("public_metrics")
    public_metrics = PublicMetrics(**raw_metrics) if raw_metrics else None

    post = Post(
        post_id=int(item["id"]),
        author_id=int(author_id),
        username=username,
        created_at=datetime.fromisoformat(item["created_at"].replace("Z", "+00:00")),
        text=text,
        media=media_objs,
        lang=lang,
        public_metrics=public_metrics,
        author_verified_type=author_verified_type,
        author_parody=author_parody,
        author_public_metrics=author_public_metrics,
    )
    return post


def _parse_posts_eligible_response(resp: dict, feed: str) -> list[PostWithContext]:
    includes_media = resp.get("includes", {}).get("media", [])
    media_by_key = {m["media_key"]: m for m in includes_media}

    includes_posts = resp.get("includes", {}).get("tweets", [])
    posts_by_id = {t["id"]: t for t in includes_posts}

    includes_users = resp.get("includes", {}).get("users", [])
    users_by_id = {u["id"]: u for u in includes_users}

    for media_obj in media_by_key.values():
        media_obj["media_type"] = media_obj.pop("type")

    posts: List[PostWithContext] = []
    for item in resp.get("data", []):
        post = _parse_individual_post(item, media_by_key, users_by_id)

        quoted_post = None
        in_reply_to_post = None
        retweeted_post = None
        if "referenced_tweets" in item:
            for ref in item["referenced_tweets"]:
                referenced_post_id = ref["id"]
                if referenced_post_id not in posts_by_id:
                    continue
                referenced_post_item = posts_by_id[referenced_post_id]
                referenced_post = _parse_individual_post(
                    referenced_post_item, media_by_key, users_by_id
                )

                if ref["type"] == "quoted":
                    assert quoted_post is None, (
                        "Multiple quoted posts found in a single post"
                    )
                    quoted_post = referenced_post
                elif ref["type"] == "replied_to":
                    assert in_reply_to_post is None, (
                        "Multiple in-reply-to posts found in a single post"
                    )
                    in_reply_to_post = referenced_post
                elif ref["type"] == "retweeted":
                    assert retweeted_post is None, (
                        "Multiple retweeted posts found in a single post"
                    )
                    retweeted_post = referenced_post
                else:
                    raise ValueError(
                        f"Unknown referenced tweet type: {ref['type']} (expected 'quoted', 'replied_to', or 'retweeted')"
                    )

        suggested_sources: list[SuggestedSource] = []
        for source in item.get("suggested_source_links_with_counts") or []:
            suggested_sources.append(
                SuggestedSource(
                    link=source["url"],
                    count=source["count"],
                )
            )
        suggested_sources.sort(key=lambda x: (-x.count, x.link))

        note_request_suggestions: list[str] = []
        seen_suggestions: set[str] = set()
        for request in item.get("note_request_suggestions") or []:
            text = (request.get("suggestion") or "").strip()
            if not text or text in seen_suggestions:
                continue
            seen_suggestions.add(text)
            note_request_suggestions.append(text)

        post_with_context = PostWithContext(
            post=post,
            quoted_post=quoted_post,
            in_reply_to_post=in_reply_to_post,
            retweeted_post=retweeted_post,
            api_feed=feed,
            suggested_sources=suggested_sources,
            note_request_suggestions=note_request_suggestions,
        )
        posts.append(post_with_context)

    return posts


def _extract_pagination_timestamp(token: str) -> int:
    decoded = json.loads(base64.b64decode(token))
    return -int(decoded["startAt"]["lkey"][0])


def fetch_feed(
    oauth: OAuth1Session,
    feed_name: str,
    post_selection: str,
    cutoff: int | None = None,
    max_results_per_page: int = 100,
    max_retries: int = 3,
    retry_backoff_seconds: float = 5.0,
) -> FeedFetchResult:
    all_posts: list[PostWithContext] = []
    unique_post_ids: set[int] = set()
    pagination_token: str | None = None
    pages_fetched = 0
    error_count = 0
    retries_remaining = max_retries
    oldest_timestamp_ms: int | None = None
    t0 = time.monotonic()

    while True:
        try:
            response = _fetch_posts_eligible_for_notes(
                oauth,
                post_selection=post_selection,
                max_results=max_results_per_page,
                pagination_token=pagination_token,
            )
        except Exception:
            error_count += 1
            if retries_remaining > 0:
                retries_remaining -= 1
                time.sleep(retry_backoff_seconds)
                continue
            else:
                elapsed = time.monotonic() - t0
                raise PostRetrievalError(
                    status_code=0,
                    message=(
                        f"Feed {feed_name}: retry budget exhausted after "
                        f"{error_count} errors, {len(unique_post_ids)} unique "
                        f"posts retrieved in {elapsed:.1f}s"
                    ),
                )

        posts = _parse_posts_eligible_response(response, feed=feed_name)
        for p in posts:
            pid = p.post.post_id
            if pid not in unique_post_ids:
                unique_post_ids.add(pid)
                all_posts.append(p)
        pages_fetched += 1

        next_token = response.get("meta", {}).get("next_token")
        if not next_token:
            elapsed = time.monotonic() - t0
            raise PostRetrievalError(
                status_code=0,
                message=(
                    f"Feed {feed_name}: no pagination token returned after "
                    f"page {pages_fetched} ({len(unique_post_ids)} unique posts, "
                    f"{elapsed:.1f}s)"
                ),
            )

        oldest_timestamp_ms = _extract_pagination_timestamp(next_token)

        if cutoff is not None and oldest_timestamp_ms < cutoff:
            break
        if cutoff is None:
            break

        pagination_token = next_token

    elapsed = time.monotonic() - t0

    return FeedFetchResult(
        posts=all_posts,
        pages_fetched=pages_fetched,
        error_count=error_count,
        elapsed_seconds=elapsed,
        oldest_timestamp_ms=oldest_timestamp_ms,
    )
