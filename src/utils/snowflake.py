def get_timestamp_from_snowflake(snowflake: int) -> int:
    return (snowflake >> 22) + 1288834974657
