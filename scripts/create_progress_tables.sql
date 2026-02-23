-- Run this in the Supabase SQL Editor

create table if not exists user_paper_progress (
    id serial primary key,
    user_id uuid references auth.users(id) on delete cascade,
    paper_number integer references papers(number) on delete cascade,
    questions_answered integer default 0,
    questions_correct integer default 0,
    points integer default 0,
    completed boolean default false,
    updated_at timestamptz default now(),
    unique(user_id, paper_number)
);

alter table user_paper_progress enable row level security;

create policy "Users can read own progress"
    on user_paper_progress for select using (auth.uid() = user_id);
create policy "Users can insert own progress"
    on user_paper_progress for insert with check (auth.uid() = user_id);
create policy "Users can update own progress"
    on user_paper_progress for update using (auth.uid() = user_id);

-- Also allow public read for leaderboard (read all)
create policy "Progress is publicly readable for leaderboard"
    on user_paper_progress for select using (true);
