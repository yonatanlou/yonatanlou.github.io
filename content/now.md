---
permalink: now.html
eleventyExcludeFromCollections: true
year2024: |
  
  - Reserve Duty.

  - Started my Masters thesis under the supervision of [Roded Sharan](https://www.cs.tau.ac.il/~roded/).

  - Bachelor party in Prague.

  - Got married to my love of my life.

  - Honeymoon in Japan and Bali

  - Moved to a new place (still Tel aviv).

  - Places: Czech Republic (Prague), Italy (Tuscany, Rome, Florence), Japan (Tokyo, Osaka, Kyoto, Hakone), Bali.



year2023: |

  - We got [Mailo](https://github.com/yonatanlou/yonatanlou.github.io/blob/main/content/img/mailo_cartoon.png?raw=true). 

  - Ive proposed to Babsi.

  - Places: Egypt (Sinai), Cyprus (Aya Napa), Netherlands (Amsterdam), Hungary (Budapest), USA (NYC, NC, CT).
  
  - 7th of October, 2023 - Reserve Duty.

year2022: |

  - Moved to Tel Aviv.

  - Started my MSC in Statistics and Data Science at Tel Aviv University.

  - Places: Lisbon, Mexico, Guatemala, Belize, Orlando, Miami, NYC.

  - Places: Portugal (Lisbon), Mexico (East coast, San Cristobal de las Casas), Guatemala, Belize, USA (Miami, NYC.

year2021: |

  - Started my job as a researcher at Forter.

  - Places: Egypt (Sinai), France (Paris), Germany (Berlin).

year2020: |

  - Started my first job as an Analyst at the ministry of finance.

  - Corona

year2019: |

  - Moved to NYC with [Babsi](https://www.instagram.com/shanniebreda).

  - Drummer at [Baby Got Back Talk](https://www.babygotbacktalk.com/).

  - Moved to Jerusalem.

  - Started my BSc in Statistics and Data Science at The Hebrew University of Jerusalem.

  - Places: France (Paris), USA (NYC, Washington D.C), Ontario, Jordan, Sinai. 



---

<style>
.timeline {
  max-width: 800px;
  margin: 2rem auto;
}

.year-section {
  margin-bottom: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  overflow: hidden;
  background: white;
}

.year-header {
  padding: 1rem 1.5rem;
  background: #f5f5f5;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  transition: background-color 0.2s;
  user-select: none;
}

.year-header:hover {
  background: #e8e8e8;
}

.year-title {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
}

.year-icon {
  font-size: 1.2rem;
  transition: transform 0.3s;
}

.year-section.active .year-icon {
  transform: rotate(180deg);
}

.year-content {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease-out;
}

.year-section.active .year-content {
  max-height: 1000px;
  transition: max-height 0.5s ease-in;
}

.year-inner {
  padding: 1.5rem;
  line-height: 1.6;
}

.year-inner h3 {
  margin-top: 0;
  color: #333;
}

.year-inner p {
  margin: 0.5rem 0;
}

.year-inner ul {
  margin: 0.5rem 0;
  padding-left: 1.5rem;
}
</style>

## Now

I'm currently working at [Tavily](https://www.tavily.com/) as an AI Researcher, building the best web infrastructure layer for agents.

Finished my Masters in Statistics and Data Science at Tel Aviv University.

<div class="timeline">

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2024</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2024 | safe }}
    </div>
  </div>
</div>

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2023</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2023 | safe }}
    </div>
  </div>
</div>

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2022</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2022 | safe }}
    </div>
  </div>
</div>

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2021</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2021 | safe }}
    </div>
  </div>
</div>

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2020</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2020 | safe }}
    </div>
  </div>
</div>

<div class="year-section">
  <div class="year-header" onclick="toggleYear(this)">
    <h2 class="year-title">2019</h2>
    <span class="year-icon">▼</span>
  </div>
  <div class="year-content">
    <div class="year-inner">
      {{ year2019 | safe }}
    </div>
  </div>
</div>

</div>

<script>
function toggleYear(header) {
  const section = header.parentElement;
  const wasActive = section.classList.contains('active');

  // Close all sections
  document.querySelectorAll('.year-section').forEach(s => {
    s.classList.remove('active');
  });

  // Open clicked section if it wasn't active
  if (!wasActive) {
    section.classList.add('active');
  }
}
</script>
