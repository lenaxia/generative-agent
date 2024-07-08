package scratchmemory

import (
	"database/sql"
	"fmt"
	"log"
	"reflect"
	"strings"
	"time"

	"github.com/lib/pq"
	"llm-agent/configs"
	"llm-agent/internal/interfaces"
	"llm-agent/internal/types"
	"llm-agent/pkg/embeddings"
	"llm-agent/pkg/utils"
	"llm-agent/pkg/vectors"
	"github.com/rankbm25/bm25"
)

// Scratch is the module for managing the agent's short-term memory
type Scratch struct {
	db              *sql.DB
	logger          *log.Logger
	embedder        embeddings.Embedder
	vectorSimilarity vectors.VectorSimilarity
	bm25            *bm25.BM25
}


// NewScratch creates a new instance of Scratch
func NewScratch(cfg *configs.Config, logger *log.Logger) (*Scratch, error) {
	db, err := sql.Open("postgres", cfg.DatabaseURL)
	if err != nil {
		return nil, err
	}

	_, err = db.Exec(`
		CREATE TABLE IF NOT EXISTS scratch (
			id SERIAL PRIMARY KEY,
			vision_r INT NOT NULL,
			att_bandwidth INT NOT NULL,
			retention INT NOT NULL,
			curr_time TIMESTAMP,
			curr_tile TEXT,
			daily_plan_req TEXT,
			name TEXT NOT NULL,
			first_name TEXT NOT NULL,
			last_name TEXT NOT NULL,
			age INT NOT NULL,
			innate TEXT NOT NULL,
			learned TEXT NOT NULL,
			currently TEXT NOT NULL,
			lifestyle TEXT NOT NULL,
			living_area TEXT NOT NULL,
			concept_forget INT NOT NULL,
			daily_reflection_time INT NOT NULL,
			daily_reflection_size INT NOT NULL,
			overlap_reflect_th INT NOT NULL,
			kw_strg_event_reflect_th INT NOT NULL,
			kw_strg_thought_reflect_th INT NOT NULL,
			recency_w REAL NOT NULL,
			relevance_w REAL NOT NULL,
			importance_w REAL NOT NULL,
			recency_decay REAL NOT NULL,
			importance_trigger_max INT NOT NULL,
			importance_trigger_curr INT NOT NULL,
			importance_ele_n INT NOT NULL,
			thought_count INT NOT NULL,
			daily_req TEXT[] NOT NULL,
			f_daily_schedule TEXT[] NOT NULL,
			f_daily_schedule_hourly_org TEXT[] NOT NULL,
			act_address TEXT,
			act_start_time TIMESTAMP,
			act_duration INT,
			act_description TEXT,
			act_pronunciatio TEXT,
			act_event TEXT,
			act_obj_description TEXT,
			act_obj_pronunciatio TEXT,
			act_obj_event TEXT,
			chatting_with TEXT,
			chat TEXT[],
			chatting_with_buffer TEXT[],
			chatting_end_time TIMESTAMP,
			act_path_set BOOLEAN NOT NULL,
			planned_path TEXT[],
			embedding VECTOR(512)
		);
	`)
	if err != nil {
		return nil, err
	}

	embedder, err := embeddings.NewEmbedder(cfg.EmbeddingConfig)
	if err != nil {
		return nil, err
	}

	vectorSimilarity, err := vectors.NewVectorSimilarity(cfg.VectorSimilarityConfig)
	if err != nil {
		return nil, err
	}

	scratch := &Scratch{
		db:              db,
		logger:          logger,
		embedder:        embedder,
		vectorSimilarity: vectorSimilarity,
	}
	scratch.initBM25()

	return scratch, nil
}

// GetMetadata returns the metadata for this service
func (s *Scratch) GetMetadata() interfaces.ServiceMetadata {
	return interfaces.ServiceMetadata{
		Description: "Manages the agent's short-term memory",
		Metadata:    make(map[string]interface{}),
	}
}

// Save saves the agent's short-term memory to the database
func (s *Scratch) Save(scratch *types.Scratch) error {
	embedding, err := s.embedder.Embed(scratch.GetStrIss())
	if err != nil {
		s.logger.Printf("Error generating embedding for scratch: %v", err)
		return err
	}

	actEvent := strings.Join(scratch.ActEvent[:], " ")
	actObjEvent := strings.Join(scratch.ActObjEvent[:], " ")

	_, err = s.db.Exec(`
		INSERT INTO scratch (
			vision_r, att_bandwidth, retention, curr_time, curr_tile, daily_plan_req, name, first_name, last_name, age, innate, learned, currently, lifestyle, living_area, concept_forget, daily_reflection_time, daily_reflection_size, overlap_reflect_th, kw_strg_event_reflect_th, kw_strg_thought_reflect_th, recency_w, relevance_w, importance_w, recency_decay, importance_trigger_max, importance_trigger_curr, importance_ele_n, thought_count, daily_req, f_daily_schedule, f_daily_schedule_hourly_org, act_address, act_start_time, act_duration, act_description, act_pronunciatio, act_event, act_obj_description, act_obj_pronunciatio, act_obj_event, chatting_with, chat, chatting_with_buffer, chatting_end_time, act_path_set, planned_path, embedding
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31, $32, $33, $34, $35, $36, $37, $38, $39, $40, $41, $42, $43, $44, $45, $46, $47, $48
		) ON CONFLICT (id) DO UPDATE SET
			vision_r = $1,
			att_bandwidth = $2,
			retention = $3,
			curr_time = $4,
			curr_tile = $5,
			daily_plan_req = $6,
			name = $7,
			first_name = $8,
			last_name = $9,
			age = $10,
			innate = $11,
			learned = $12,
			currently = $13,
			lifestyle = $14,
			living_area = $15,
			concept_forget = $16,
			daily_reflection_time = $17,
			daily_reflection_size = $18,
			overlap_reflect_th = $19,
			kw_strg_event_reflect_th = $20,
			kw_strg_thought_reflect_th = $21,
			recency_w = $22,
			relevance_w = $23,
			importance_w = $24,
			recency_decay = $25, 
                        importance_trigger_max = $26,
			importance_trigger_curr = $27,
			importance_ele_n = $28,
			thought_count = $29,
			daily_req = $30,
			f_daily_schedule = $31,
			f_daily_schedule_hourly_org = $32,
			act_address = $33,
			act_start_time = $34,
			act_duration = $35,
			act_description = $36,
			act_pronunciatio = $37,
			act_event = $38,
			act_obj_description = $39,
			act_obj_pronunciatio = $40,
			act_obj_event = $41,
			chatting_with = $42,
			chat = $43,
			chatting_with_buffer = $44,
			chatting_end_time = $45,
			act_path_set = $46,
			planned_path = $47,
			embedding = $48
	`, scratch.VisionR, scratch.AttBandwidth, scratch.Retention, scratch.CurrTime, scratch.CurrTile, scratch.DailyPlanReq, scratch.Name, scratch.FirstName, scratch.LastName, scratch.Age, scratch.Innate, scratch.Learned, scratch.Currently, scratch.Lifestyle, scratch.LivingArea, scratch.ConceptForget, scratch.DailyReflectionTime, scratch.DailyReflectionSize, scratch.OverlapReflectTh, scratch.KwStrgEventReflectTh, scratch.KwStrgThoughtReflect, scratch.RecencyW, scratch.RelevanceW, scratch.ImportanceW, scratch.RecencyDecay, scratch.ImportanceTriggerMax, scratch.ImportanceTriggerCurr, scratch.ImportanceEleN, scratch.ThoughtCount, pq.Array(scratch.DailyReq), pq.Array(scratch.FDailySchedule), pq.Array(scratch.FDailyScheduleHourly), scratch.ActAddress, scratch.ActStartTime, scratch.ActDuration, scratch.ActDescription, scratch.ActPronunciation, actEvent, scratch.ActObjDescription, scratch.ActObjPronunciation, actObjEvent, scratch.ChattingWith, pq.Array(scratch.Chat), pq.Array(scratch.ChattingWithBuffer), scratch.ChattingEndTime, scratch.ActPathSet, pq.Array(scratch.PlannedPath), pq.Array(embedding))
	if err != nil {
		s.logger.Printf("Error saving scratch: %v", err)
		return err
	}

	return nil
}

// Load loads the agent's short-term memory from the database
func (s *Scratch) Load() (*types.Scratch, error) {
	var scratch types.Scratch
	row := s.db.QueryRow(`
		SELECT vision_r, att_bandwidth, retention, curr_time, curr_tile, daily_plan_req, name, first_name, last_name, age, innate, learned, currently, lifestyle, living_area, concept_forget, daily_reflection_time, daily_reflection_size, overlap_reflect_th, kw_strg_event_reflect_th, kw_strg_thought_reflect_th, recency_w, relevance_w, importance_w, recency_decay, importance_trigger_max, importance_trigger_curr, importance_ele_n, thought_count, daily_req, f_daily_schedule, f_daily_schedule_hourly_org, act_address, act_start_time, act_duration, act_description, act_pronunciatio, act_event, act_obj_description, act_obj_pronunciatio, act_obj_event, chatting_with, chat, chatting_with_buffer, chatting_end_time, act_path_set, planned_path, embedding
		FROM scratch
		ORDER BY id DESC
		LIMIT 1
	`)

	var actEventStr, actObjEventStr string
	err := row.Scan(&scratch.VisionR, &scratch.AttBandwidth, &scratch.Retention, &scratch.CurrTime, &scratch.CurrTile, &scratch.DailyPlanReq, &scratch.Name, &scratch.FirstName, &scratch.LastName, &scratch.Age, &scratch.Innate, &scratch.Learned, &scratch.Currently, &scratch.Lifestyle, &scratch.LivingArea, &scratch.ConceptForget, &scratch.DailyReflectionTime, &scratch.DailyReflectionSize, &scratch.OverlapReflectTh, &scratch.KwStrgEventReflectTh, &scratch.KwStrgThoughtReflect, &scratch.RecencyW, &scratch.RelevanceW, &scratch.ImportanceW, &scratch.RecencyDecay, &scratch.ImportanceTriggerMax, &scratch.ImportanceTriggerCurr, &scratch.ImportanceEleN, &scratch.ThoughtCount, pq.Array(&scratch.DailyReq), pq.Array(&scratch.FDailySchedule), pq.Array(&scratch.FDailyScheduleHourly), &scratch.ActAddress, &scratch.ActStartTime, &scratch.ActDuration, &scratch.ActDescription, &scratch.ActPronunciation, &actEventStr, &scratch.ActObjDescription, &scratch.ActObjPronunciation, &actObjEventStr, &scratch.ChattingWith, pq.Array(&scratch.Chat), pq.Array(&scratch.ChattingWithBuffer), &scratch.ChattingEndTime, &scratch.ActPathSet, pq.Array(&scratch.PlannedPath), pq.Array(&scratch.Embedding))
	if err != nil {
		s.logger.Printf("Error loading scratch: %v", err)
		return nil, err
	}

	scratch.ActEvent = strings.Split(actEventStr, " ")
	scratch.ActObjEvent = strings.Split(actObjEventStr, " ")

	return &scratch, nil
}

// GetStrIss returns the identity stable set summary of the agent
func (s *Scratch) GetStrIss(scratch *types.Scratch) string {
	commonset := fmt.Sprintf("Name: %s\nAge: %d\nInnate traits: %s\nLearned traits: %s\nCurrently: %s\nLifestyle: %s\nDaily plan requirement: %s\nCurrent Date: %s\n", scratch.Name, scratch.Age, scratch.Innate, scratch.Learned, scratch.Currently, scratch.Lifestyle, scratch.DailyPlanReq, scratch.CurrTime.Format("Monday January 2, 2006"))
	return commonset
}

// GetStrName returns the agent's name
func (s *Scratch) GetStrName(scratch *types.Scratch) string {
	return scratch.Name
}

// GetStrFirstname returns the agent's first name
func (s *Scratch) GetStrFirstname(scratch *types.Scratch) string {
	return scratch.FirstName
}

// GetStrLastname returns the agent's last name
func (s *Scratch) GetStrLastname(scratch *types.Scratch) string {
	return scratch.LastName
}

// GetStrAge returns the agent's age
func (s *Scratch) GetStrAge(scratch *types.Scratch) string {
	return fmt.Sprintf("%d", scratch.Age)
}

// GetStrInnate returns the agent's innate traits
func (s *Scratch) GetStrInnate(scratch *types.Scratch) string {
	return scratch.Innate
}

// GetStrLearned returns the agent's learned traits
func (s *Scratch) GetStrLearned(scratch *types.Scratch) string {
	return scratch.Learned
}

// GetStrCurrently returns the agent's current state
func (s *Scratch) GetStrCurrently(scratch *types.Scratch) string {
	return scratch.Currently
}

// GetStrLifestyle returns the agent's lifestyle
func (s *Scratch) GetStrLifestyle(scratch *types.Scratch) string {
	return scratch.Lifestyle
}

// GetStrDailyPlanReq returns the agent's daily plan requirement
func (s *Scratch) GetStrDailyPlanReq(scratch *types.Scratch) string {
	return scratch.DailyPlanReq
}

// GetStrCurrDateStr returns the current date as a string
func (s *Scratch) GetStrCurrDateStr(scratch *types.Scratch) string {
	return scratch.CurrTime.Format("Monday January 2, 2006")
}

// GetCurrEvent returns the agent's current event
func (s *Scratch) GetCurrEvent(scratch *types.Scratch) (string, string, string) {
	if scratch.ActAddress == "" {
		return scratch.Name, "", ""
	}
	return scratch.ActEvent[0], scratch.ActEvent[1], scratch.ActEvent[2]
}

// GetCurrEventAndDesc returns the agent's current event and its description
func (s *Scratch) GetCurrEventAndDesc(scratch *types.Scratch) (string, string, string, string) {
	if scratch.ActAddress == "" {
		return scratch.Name, "", "", ""
	}
	return scratch.ActEvent[0], scratch.ActEvent[1], scratch.ActEvent[2], scratch.ActDescription
}

// GetCurrObjEventAndDesc returns the agent's current object event and its description
func (s *Scratch) GetCurrObjEventAndDesc(scratch *types.Scratch) (string, string, string, string) {
	if scratch.ActAddress == "" {
		return "", "", "", ""
	}
	return scratch.ActAddress, scratch.ActObjEvent[0], scratch.ActObjEvent[1], scratch.ActObjEvent[2], scratch.ActObjDescription
}

// AddNewAction adds a new action to the agent's short-term memory
func (s *Scratch) AddNewAction(scratch *types.Scratch, actionAddress, actionDescription, actionPronunciatio string, actionDuration int, actionEvent, actObjDescription, actObjPronunciatio [3]string, chattingWith string, chat [][]string, chattingWithBuffer []string, chattingEndTime time.Time) {
	scratch.ActAddress = actionAddress
	scratch.ActDescription = actionDescription
	scratch.ActPronunciation = actionPronunciatio
	scratch.ActDuration = actionDuration
	scratch.ActEvent = actionEvent
	scratch.ActObjDescription = actObjDescription
	scratch.ActObjPronunciation = actObjPronunciatio
	scratch.ActObjEvent = actObjEvent
	scratch.ChattingWith = chattingWith
	scratch.Chat = chat
	scratch.ChattingWithBuffer = chattingWithBuffer
	scratch.ChattingEndTime = &chattingEndTime
	scratch.ActStartTime = &time.Now()
	scratch.ActPathSet = false
}

// ActTimeStr returns the current time as a string
func (s *Scratch) ActTimeStr(scratch *types.Scratch) string {
	return scratch.ActStartTime.Format("15:04 PM")
}

// ActCheckFinished checks if the current action has finished
func (s *Scratch) ActCheckFinished(scratch *types.Scratch) bool {
	if scratch.ActAddress == "" {
		return true
	}

	var endTime time.Time
	if scratch.ChattingWith != "" {
		endTime = *scratch.ChattingEndTime
	} else {
		x := *scratch.ActStartTime
		if x.Second() != 0 {
			x = x.Add(-time.Duration(x.Second()) * time.Second)
			x = x.Add(time.Minute)
		}
		endTime = x.Add(time.Duration(scratch.ActDuration) * time.Minute)
	}

	if endTime.Format("15:04:05") == scratch.CurrTime.Format("15:04:05") {
		return true
	}
	return false
}

// ActSummarize summarizes the current action as a dictionary
func (s *Scratch) ActSummarize(scratch *types.Scratch) map[string]interface{} {
	exp := make(map[string]interface{})
	exp["persona"] = scratch.Name
	exp["address"] = scratch.ActAddress
	exp["start_datetime"] = scratch.ActStartTime
	exp["duration"] = scratch.ActDuration
	exp["description"] = scratch.ActDescription
	exp["pronunciatio"] = scratch.ActPronunciation
	return exp
}

// ActSummaryStr returns a string summary of the current action
func (s *Scratch) ActSummaryStr(scratch *types.Scratch) string {
	startDatetimeStr := scratch.ActStartTime.Format("Monday January 2, 2006 -- 3:04 PM")
	ret := fmt.Sprintf("[%s]\n", startDatetimeStr)
	ret += fmt.Sprintf("Activity: %s is %s\n", scratch.Name, scratch.ActDescription)
	ret += fmt.Sprintf("Address: %s\n", scratch.ActAddress)
	ret += fmt.Sprintf("Duration in minutes (e.g., x min): %d min\n", scratch.ActDuration)
	return ret
}

// GetStrDailyScheduleSummary returns a string summary of the daily schedule
func (s *Scratch) GetStrDailyScheduleSummary(scratch *types.Scratch) string {
	ret := ""
	currMinSum := 0
	for _, row := range scratch.FDailySchedule {
		currMinSum += row[1]
		hour := currMinSum / 60
		minute := currMinSum % 60
		ret += fmt.Sprintf("%02d:%02d || %s\n", hour, minute, row[0])
	}
	return ret
}

// GetStrDailyScheduleHourlyOrgSummary returns a string summary of the original hourly daily schedule
func (s *Scratch) GetStrDailyScheduleHourlyOrgSummary(scratch *types.Scratch) string {
	ret := ""
	currMinSum := 0
	for _, row := range scratch.FDailyScheduleHourly {
		currMinSum += row[1]
		hour := currMinSum / 60
		minute := currMinSum % 60
		ret += fmt.Sprintf("%02d:%02d || %s\n", hour, minute, row[0])
	}
	return ret
}

// RetrieveScratchesByFilter retrieves scratches from the database based on filters
func (s *Scratch) RetrieveScratchesByFilter(filters ...FilterScratch) ([]types.Scratch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var scratches []types.Scratch
	query := `SELECT vision_r, att_bandwidth, retention, curr_time, curr_tile, daily_plan_req, name, first_name, last_name, age, innate, learned, currently, lifestyle, living_area, concept_forget, daily_reflection_time, daily_reflection_size, overlap_reflect_th, kw_strg_event_reflect_th, kw_strg_thought_reflect_th, recency_w, relevance_w, importance_w, recency_decay, importance_trigger_max, importance_trigger_curr, importance_ele_n, thought_count, daily_req, f_daily_schedule, f_daily_schedule_hourly_org, act_address, act_start_time, act_duration, act_description, act_pronunciatio, act_event, act_obj_description, act_obj_pronunciatio, act_obj_event, chatting_with, chat, chatting_with_buffer, chatting_end_time, act_path_set, planned_path, embedding
		FROM scratch`

	// Apply filters to the query
	for _, filter := range filters {
		if filter.Name != "" {
			query += fmt.Sprintf(" AND name = '%s'", filter.Name)
		}
		if filter.FirstName != "" {
			query += fmt.Sprintf(" AND first_name = '%s'", filter.FirstName)
		}
		if filter.LastName != "" {
			query += fmt.Sprintf(" AND last_name = '%s'", filter.LastName)
		}
		if filter.Age != 0 {
			query += fmt.Sprintf(" AND age = %d", filter.Age)
		}
		if filter.Innate != "" {
			query += fmt.Sprintf(" AND innate = '%s'", filter.Innate)
		}
		if filter.Learned != "" {
			query += fmt.Sprintf(" AND learned = '%s'", filter.Learned)
		}
		if filter.Currently != "" {
			query += fmt.Sprintf(" AND currently = '%s'", filter.Currently)
		}
		if filter.Lifestyle != "" {
			query += fmt.Sprintf(" AND lifestyle = '%s'", filter.Lifestyle)
		}
		if filter.LivingArea != "" {
			query += fmt.Sprintf(" AND living_area = '%s'", filter.LivingArea)
		}
		if filter.DailyPlanReq != "" {
			query += fmt.Sprintf(" AND daily_plan_req = '%s'", filter.DailyPlanReq)
		}
		if filter.ActDescription != "" {
			query += fmt.Sprintf(" AND act_description = '%s'", filter.ActDescription)
		}
		if len(filter.ActEvent) == 3 {
			actEvent := strings.Join(filter.ActEvent[:], " ")
			query += fmt.Sprintf(" AND act_event = '%s'", actEvent)
		}
		if len(filter.ActObjEvent) == 3 {
			actObjEvent := strings.Join(filter.ActObjEvent[:], " ")
			query += fmt.Sprintf(" AND act_obj_event = '%s'", actObjEvent)
		}
		if filter.PlaintextQuery != "" {
			queryTokens := utils.Tokenize(filter.PlaintextQuery)
			scores := s.bm25.ScoreTokens(queryTokens)
			query += fmt.Sprintf(" AND (SELECT SUM(bm25_score(keywords, %s, %s)) FROM unnest(keywords) AS keyword) > 0", pq.Array(queryTokens), pq.Array(scores))
		}
		if !filter.StartTime.IsZero() {
			query += fmt.Sprintf(" AND curr_time >= '%s'", filter.StartTime.Format(time.RFC3339))
		}
		if !filter.EndTime.IsZero() {
			query += fmt.Sprintf(" AND curr_time <= '%s'", filter.EndTime.Format(time.RFC3339))
		}
		if len(filter.EmbeddingIDs) > 0 {
			query += fmt.Sprintf(" AND embedding_id IN (%s)", pq.Array(filter.EmbeddingIDs))
		}
	}

	rows, err := s.db.Query(query)
	if err != nil {
		s.logger.Printf("Error retrieving scratches: %v", err)
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var scratch types.Scratch
		var actEventStr, actObjEventStr string
		err = rows.Scan(&scratch.VisionR, &scratch.AttBandwidth, &scratch.Retention, &scratch.CurrTime, &scratch.CurrTile, &scratch.DailyPlanReq, &scratch.Name, &scratch.FirstName, &scratch.LastName, &scratch.Age, &scratch.Innate, &scratch.Learned, &scratch.Currently, &scratch.Lifestyle, &scratch.LivingArea, &scratch.ConceptForget, &scratch.DailyReflectionTime, &scratch.DailyReflectionSize, &scratch.OverlapReflectTh, &scratch.KwStrgEventReflectTh, &scratch.KwStrgThoughtReflect, &scratch.RecencyW, &scratch.RelevanceW, &scratch.ImportanceW, &scratch.RecencyDecay, &scratch.ImportanceTriggerMax, &scratch.ImportanceTriggerCurr, &scratch.ImportanceEleN, &scratch.ThoughtCount, pq.Array(&scratch.DailyReq), pq.Array(&scratch.FDailySchedule), pq.Array(&scratch.FDailyScheduleHourly), &scratch.ActAddress, &scratch.ActStartTime, &scratch.ActDuration, &scratch.ActDescription, &scratch.ActPronunciation, &actEventStr, &scratch.ActObjDescription, &scratch.ActObjPronunciation, &actObjEventStr, &scratch.ChattingWith, pq.Array(&scratch.Chat), pq.Array(&scratch.ChattingWithBuffer), &scratch.ChattingEndTime, &scratch.ActPathSet, pq.Array(&scratch.PlannedPath), pq.Array(&scratch.Embedding))
		if err != nil {
			s.logger.Printf("Error scanning scratch: %v", err)
			return nil, err
		}
		scratch.ActEvent = strings.Split(actEventStr, " ")
		scratch.ActObjEvent = strings.Split(actObjEventStr, " ")
		scratches = append(scratches, scratch)
	}

	return scratches, nil
}

// RetrieveScratchesByVector retrieves scratches from the database based on vector similarity
func (s *Scratch) RetrieveScratchesByVector(queryEmbedding []float32, k int) ([]types.Scratch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var scratches []types.Scratch
	query := `
		SELECT vision_r, att_bandwidth, retention, curr_time, curr_tile, daily_plan_req, name, first_name, last_name, age, innate, learned, currently, lifestyle, living_area, concept_forget, daily_reflection_time, daily_reflection_size, overlap_reflect_th, kw_strg_event_reflect_th, kw_strg_thought_reflect_th, recency_w, relevance_w, importance_w, recency_decay, importance_trigger_max, importance_trigger_curr, importance_ele_n, thought_count, daily_req, f_daily_schedule, f_daily_schedule_hourly_org, act_address, act_start_time, act_duration, act_description, act_pronunciatio, act_event, act_obj_description, act_obj_pronunciatio, act_obj_event, chatting_with, chat, chatting_with_buffer, chatting_end_time, act_path_set, planned_path
		FROM scratch
		ORDER BY embedding <-> $1 ASC
		LIMIT $2
	`

	rows, err := s.db.Query(query, pq.Array(queryEmbedding), k)
	if err != nil {
		s.logger.Printf("Error retrieving scratches by vector: %v", err)
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var scratch types.Scratch
		var actEventStr, actObjEventStr string
		err = rows.Scan(&scratch.VisionR, &scratch.AttBandwidth, &scratch.Retention, &scratch.CurrTime, &scratch.CurrTile, &scratch.DailyPlanReq, &scratch.Name, &scratch.FirstName, &scratch.LastName, &scratch.Age, &scratch.Innate, &scratch.Learned, &scratch.Currently, &scratch.Lifestyle, &scratch.LivingArea, &scratch.ConceptForget, &scratch.DailyReflectionTime, &scratch.DailyReflectionSize, &scratch.OverlapReflectTh, &scratch.KwStrgEventReflectTh, &scratch.KwStrgThoughtReflect, &scratch.RecencyW, &scratch.RelevanceW, &scratch.ImportanceW, &scratch.RecencyDecay, &scratch.ImportanceTriggerMax, &scratch.ImportanceTriggerCurr, &scratch.ImportanceEleN, &scratch.ThoughtCount, pq.Array(&scratch.DailyReq), pq.Array(&scratch.FDailySchedule), pq.Array(&scratch.FDailyScheduleHourly), &scratch.ActAddress, &scratch.ActStartTime, &scratch.ActDuration, &scratch.ActDescription, &scratch.ActPronunciation, &actEventStr, &scratch.ActObjDescription, &scratch.ActObjPronunciation, &actObjEventStr, &scratch.ChattingWith, pq.Array(&scratch.Chat), pq.Array(&scratch.ChattingWithBuffer), &scratch.ChattingEndTime, &scratch.ActPathSet, pq.Array(&scratch.PlannedPath))
		if err != nil {
			s.logger.Printf("Error scanning scratch: %v", err)
			return nil, err
		}
		scratch.ActEvent = strings.Split(actEventStr, " ")
		scratch.ActObjEvent = strings.Split(actObjEventStr, " ")
		scratches = append(scratches, scratch)
	}

	return scratches, nil
}

func (s *Scratch) initBM25() {
	var corpus []string
	rows, err := s.db.Query(`SELECT act_description FROM scratch`)
	if err != nil {
		s.logger.Printf("Error retrieving corpus for BM25: %v", err)
		return
	}
	defer rows.Close()

	for rows.Next() {
		var description string
		err = rows.Scan(&description)
		if err != nil {
			s.logger.Printf("Error scanning description for BM25: %v", err)
			return
		}
		corpus = append(corpus, description)
	}

	s.bm25 = bm25.NewBM25(corpus)
}
