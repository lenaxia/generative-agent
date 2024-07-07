module llm-agent

go 1.16

require (
    github.com/fsnotify/fsnotify v1.6.0
    github.com/lenaxia/llm-agent/configs v0.0.0
    github.com/lenaxia/llm-agent/internal/interfaces v0.0.0
    github.com/lenaxia/llm-agent/pkg/utils v0.0.0
    github.com/lib/pq v1.10.7
    github.com/rankbm25/bm25 v0.0.0-20220905184626-2b9e3a1d0b4e
)

replace (
    github.com/lenaxia/llm-agent/configs => ./configs
    github.com/lenaxia/llm-agent/internal/interfaces => ./internal/interfaces
    github.com/lenaxia/llm-agent/pkg/utils => ./pkg/utils
)
