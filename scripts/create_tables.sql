-- Run this in the Supabase SQL Editor

create table if not exists papers (
    id serial primary key,
    number integer unique not null,
    slug text unique not null,
    title text not null,
    authors text,
    year integer,
    category text not null,
    pdf_filename text,
    link text,
    summary text,
    created_at timestamptz default now()
);

-- Allow public read access
alter table papers enable row level security;
create policy "Papers are publicly readable"
    on papers for select
    using (true);
