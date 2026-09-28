from typing import Literal

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel


MediaType = Literal["animated_gif", "photo", "video"]
RejectorStatus = Literal["PASS", "ERROR", "REJECT"]


class TestResult(BaseModel):
    evaluator_score_bucket: str
    evaluator_type: str


class TagCount(BaseModel):
    tag_name: str
    tag_count: int


class FactorBucketCounts(BaseModel):
    helpful_count: int = 0
    not_helpful_count: int = 0
    somewhat_helpful_count: int = 0
    helpful_tag_counts: list[TagCount] = []
    not_helpful_tag_counts: list[TagCount] = []


class NoteRatings(BaseModel):
    negative: FactorBucketCounts
    neutral: FactorBucketCounts
    positive: FactorBucketCounts


class NoteStatus(BaseModel):
    note_id: int
    post_id: int
    status: str
    test_result: List[TestResult] | None = None
    note_text: str | None = None
    ratings: NoteRatings | None = None
    submitter: str | None = None


class Media(BaseModel):
    media_key: str
    media_type: MediaType
    url: str | None = None
    preview_image_url: str | None = None
    height: int | None = None
    width: int | None = None
    duration_ms: int | None = None
    view_count: int | None = None


class PublicMetrics(BaseModel):
    retweet_count: int = 0
    reply_count: int = 0
    like_count: int = 0
    quote_count: int = 0
    bookmark_count: int = 0
    impression_count: int = 0


class UserPublicMetrics(BaseModel):
    followers_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    listed_count: int = 0
    like_count: int = 0
    media_count: int = 0


class Post(BaseModel):
    post_id: int
    author_id: int
    username: str
    created_at: datetime
    text: str
    media: List[Media]
    lang: str | None = None
    public_metrics: PublicMetrics | None = None
    author_verified_type: str | None = None
    author_parody: bool | None = None
    author_public_metrics: UserPublicMetrics | None = None


class SuggestedSource(BaseModel):
    link: str
    count: int


class AuthorHistory(BaseModel):
    hist_note_count: int
    hist_crh_count: int
    hist_crnh_count: int
    hist_total_ratings: int


class PostWithContext(BaseModel):
    post: Post
    quoted_post: Post | None = None
    in_reply_to_post: Post | None = None
    retweeted_post: Post | None = None
    api_feed: str
    timed_feed: str | None = None
    suggested_sources: List[SuggestedSource] = []

    note_request_suggestions: List[str] = []
    enqueued_at: int | None = None
    notable_post_prediction: float | None = None
    author_history: AuthorHistory | None = None


class GrokOutput(BaseModel):
    content: str | None = None
    tool_calls: list[dict] | None = None
    citations: list[str] | None = None
    parsed_trace: str | None = None
    final_prompt: str | None = None
    final_reasoning_content: str | None = None
    final_response_content: str | None = None
    model: str | None = None
    response_id: str | None = None
    latency: float | None = None


class ProposedNote(BaseModel):
    post_id: int
    note_text: str


class MisleadingTag(str, Enum):
    factual_error = "factual_error"
    manipulated_media = "manipulated_media"
    outdated_information = "outdated_information"
    missing_important_context = "missing_important_context"
    disputed_claim_as_fact = "disputed_claim_as_fact"
    misinterpreted_satire = "misinterpreted_satire"
    other = "other"


class ProposedMisleadingNote(ProposedNote):
    misleading_tags: List[MisleadingTag]


class NoteResult(BaseModel):
    post: PostWithContext
    writer_name: str
    writing_prompt: str

    attempt_id: int

    grok_output: GrokOutput | None = None

    note: ProposedMisleadingNote | None = None
    unsupported_media_types: set[MediaType] | None = None
    refusal: str | None = None
    error: str | None = None

    over_length_note: str | None = None

    co_score: float | None = None
    co_threshold: float | None = None


class RejectorSampleResult(BaseModel):
    score: float | None = None
    reasoning: str | None = None
    error: str | None = None
    model: str | None = None
    response_id: str | None = None
    latency: float | None = None


class RejectorResult(BaseModel):
    status: RejectorStatus
    mean_score: float | None = None
    sample_results: list[RejectorSampleResult]

    n_crh_examples: int | None = None
    n_crnh_examples: int | None = None
    n_deleted_as_crnh: int | None = None
    crh_note_id_range: tuple[int, int] | None = None
    crnh_note_id_range: tuple[int, int] | None = None


class ScreenshotRejectorResult(BaseModel):
    status: RejectorStatus
    mean_score: float | None = None
    sample_results: list[RejectorSampleResult]
    screenshot_files: list[str]


@dataclass
class ScreenshotResult:
    file_prefix: str
    url: str
    success: bool
    screenshots_saved: list[str] = field(default_factory=list)
    error_message: str | None = None
