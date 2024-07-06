package types

// Scratch represents the short-term memory of the agent
type Scratch struct {
	VisionR              int           `json:"vision_r"`
	AttBandwidth         int           `json:"att_bandwidth"`
	Retention            int           `json:"retention"`
	CurrTime             *time.Time    `json:"curr_time,omitempty"`
	CurrTile             string        `json:"curr_tile"`
	DailyPlanReq         string        `json:"daily_plan_req"`
	Name                 string        `json:"name"`
	FirstName            string        `json:"first_name"`
	LastName             string        `json:"last_name"`
	Age                  int           `json:"age"`
	Innate               string        `json:"innate"`
	Learned              string        `json:"learned"`
	Currently            string        `json:"currently"`
	Lifestyle            string        `json:"lifestyle"`
	LivingArea           string        `json:"living_area"`
	ConceptForget        int           `json:"concept_forget"`
	DailyReflectionTime  int           `json:"daily_reflection_time"`
	DailyReflectionSize  int           `json:"daily_reflection_size"`
	OverlapReflectTh     int           `json:"overlap_reflect_th"`
	KwStrgEventReflectTh int           `json:"kw_strg_event_reflect_th"`
	KwStrgThoughtReflect int           `json:"kw_strg_thought_reflect_th"`
	RecencyW             float32       `json:"recency_w"`
	RelevanceW           float32       `json:"relevance_w"`
	ImportanceW          float32       `json:"importance_w"`
	RecencyDecay         float32       `json:"recency_decay"`
	ImportanceTriggerMax int           `json:"importance_trigger_max"`
	ImportanceTriggerCurr int          `json:"importance_trigger_curr"`
	ImportanceEleN       int           `json:"importance_ele_n"`
	ThoughtCount         int           `json:"thought_count"`
	DailyReq             []string      `json:"daily_req"`
	FDailySchedule       [][]string    `json:"f_daily_schedule"`
	FDailyScheduleHourly [][]string    `json:"f_daily_schedule_hourly_org"`
	ActAddress           string        `json:"act_address"`
	ActStartTime         *time.Time    `json:"act_start_time,omitempty"`
	ActDuration          int           `json:"act_duration"`
	ActDescription       string        `json:"act_description"`
	ActPronunciation     string        `json:"act_pronunciatio"`
	ActEvent             [3]string     `json:"act_event"`
	ActObjDescription    string        `json:"act_obj_description"`
	ActObjPronunciation  string        `json:"act_obj_pronunciatio"`
	ActObjEvent          [3]string     `json:"act_obj_event"`
	ChattingWith         string        `json:"chatting_with"`
	Chat                 [][]string    `json:"chat"`
	ChattingWithBuffer   []string      `json:"chatting_with_buffer"`
	ChattingEndTime      *time.Time    `json:"chatting_end_time,omitempty"`
	ActPathSet           bool          `json:"act_path_set"`
	PlannedPath          []string      `json:"planned_path"`
}
