-- Run this in the Supabase SQL Editor

create table if not exists profiles (
    id uuid primary key references auth.users(id) on delete cascade,
    username text unique not null,
    display_name text,
    created_at timestamptz default now()
);

alter table profiles enable row level security;

-- Anyone can read profiles
create policy "Profiles are publicly readable"
    on profiles for select
    using (true);

-- Users can update their own profile
create policy "Users can update own profile"
    on profiles for update
    using (auth.uid() = id);

-- Users can insert their own profile
create policy "Users can insert own profile"
    on profiles for insert
    with check (auth.uid() = id);
