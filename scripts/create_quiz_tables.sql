-- Run this in the Supabase SQL Editor

create table if not exists questions (
    id serial primary key,
    paper_number integer references papers(number) on delete cascade,
    question text not null,
    choices jsonb not null,        -- ["choice A", "choice B", "choice C", "choice D"]
    correct_index integer not null, -- 0-3
    explanation text,
    created_at timestamptz default now()
);

create table if not exists quiz_attempts (
    id serial primary key,
    user_id uuid references auth.users(id) on delete cascade,
    question_id integer references questions(id) on delete cascade,
    selected_index integer not null,
    is_correct boolean not null,
    created_at timestamptz default now()
);

-- RLS
alter table questions enable row level security;
create policy "Questions are publicly readable"
    on questions for select using (true);

alter table quiz_attempts enable row level security;
create policy "Users can read own attempts"
    on quiz_attempts for select using (auth.uid() = user_id);
create policy "Users can insert own attempts"
    on quiz_attempts for insert with check (auth.uid() = user_id);
