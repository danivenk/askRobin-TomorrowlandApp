// this snippet is made for https://festivalviewer.com/tomorrowland/lineup/2024

// class to contain artists' performances with name, stage, host, tmie, date, year weekday, weekend & genre
class Artist {
    constructor(data) {
        data = Array.from(data.querySelectorAll("td")).map(x => x.innerText);
        let time;
        [this.name, this.stage, this.host, time, this.date,
         this.year, this.weekday, this.weekend, this.genre] = data;

        [this.time_start, this.time_end] = time.split("-").map(x => x.trim());
    }
}

// retrieve the table data
let artists = []
data = document.querySelectorAll("#table_id tr.even, #table_id tr.odd");
header = document.querySelector("#table_id tr");
// console.log(header.innerText);

// parse the data
for (let dat of data) {
    artists.push(new Artist(dat));
}

// log as JSON
console.log(JSON.stringify(artists))