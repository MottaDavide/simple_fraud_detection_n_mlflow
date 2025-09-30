drop table stats_table;

create table stats_table (
       day_num      number,
       challenge_id      number,
       total_views       number,
       total_unique_views number
);

drop table submission_stats_table;

create table submission_stats_table (
       day_num      number,
       challenge_id      number,
       total_submissions       number,
       total_accepted_submissions number
);

truncate table stats_table;

insert into stats_table values(1,47127, 26, 19);
insert into stats_table values(2,47127, 15, 14);
insert into stats_table values(1,18765, 43, 10);
insert into stats_table values(2,18765, 72, 13);
insert into stats_table values(3,75516, 35, 17);
insert into stats_table values(1,60292, 11, 10);
insert into stats_table values(3,72974, 41, 15);
insert into stats_table values(4,75516, 75, 11);


truncate table submission_stats_table;

insert into submission_stats_table values(1, 75516, 34, 12);
insert into submission_stats_table values(1, 47127, 27, 10);
insert into submission_stats_table values(2, 47127, 56, 18);
insert into submission_stats_table values(2, 75516, 74, 12);
insert into submission_stats_table values(3, 75516, 83, 8);
insert into submission_stats_table values(3, 72974, 68, 24);
insert into submission_stats_table values(5, 72974, 82, 14);
insert into submission_stats_table values(3, 47127, 28, 11);

commit;

-----

select *
from stats_table;

select *
from submission_stats_table;
