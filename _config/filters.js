import { DateTime } from "luxon";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export default function(eleventyConfig) {
	eleventyConfig.addFilter("readableDate", (dateObj, format, zone) => {
		// Formatting tokens for Luxon: https://moment.github.io/luxon/#/formatting?id=table-of-tokens
		return DateTime.fromJSDate(dateObj, { zone: zone || "utc" }).toFormat(format || "dd LLLL yyyy");
	});

	eleventyConfig.addFilter("htmlDateString", (dateObj) => {
		// dateObj input: https://html.spec.whatwg.org/multipage/common-microsyntaxes.html#valid-date-string
		return DateTime.fromJSDate(dateObj, { zone: "utc" }).toFormat('yyyy-LL-dd');
	});

	// Get the first `n` elements of a collection.
	eleventyConfig.addFilter("head", (array, n) => {
		if(!Array.isArray(array) || array.length === 0) {
			return [];
		}
		if( n < 0 ) {
			return array.slice(n);
		}

		return array.slice(0, n);
	});

	// Return the smallest number argument
	eleventyConfig.addFilter("min", (...numbers) => {
		return Math.min.apply(null, numbers);
	});

	// Return the keys used in an object
	eleventyConfig.addFilter("getKeys", target => {
		return Object.keys(target);
	});

	eleventyConfig.addFilter("filterTagList", function filterTagList(tags) {
		return (tags || []).filter(tag => ["all", "posts"].indexOf(tag) === -1);
	});

	// Parse OPML file and generate blog links
	eleventyConfig.addFilter("generateBlogLinks", function() {
		try {
			const opmlPath = path.resolve('content/files/subscriptions-2025-07-12');
			const opmlContent = fs.readFileSync(opmlPath, 'utf8');
			console.log('OPML Content:', opmlContent); // Debug
			// Regex: match <outline ... xmlUrl="..." ... title="..." ... />
			const outlineMatches = opmlContent.match(/<outline[^>]*xmlUrl="([^"]*)"[^>]*title="([^"]*)"[^>]*\/>/g);
			console.log('Outline Matches:', outlineMatches); // Debug
			if (!outlineMatches) {
				return "<!-- No blog subscriptions found -->";
			}
			const blogs = [];
			outlineMatches.forEach(match => {
				const urlMatch = match.match(/xmlUrl="([^"]*)"/);
				const titleMatch = match.match(/title="([^"]*)"/);
				if (titleMatch && urlMatch) {
					const title = titleMatch[1];
					const feedUrl = urlMatch[1];
					let blogUrl = feedUrl;
					if (feedUrl.includes('/feed/') || feedUrl.includes('/atom/') || feedUrl.includes('/rss') || feedUrl.includes('.xml')) {
						blogUrl = feedUrl.replace(/\/feed\/.*$/, '')
							.replace(/\/atom\/.*$/, '')
							.replace(/\/rss.*$/, '')
							.replace(/\.xml$/, '')
							.replace(/\/$/, '');
					}
					if (!blogUrl.includes('yonatanlou.github.io')) {
						blogs.push({ title, url: blogUrl });
					}
				}
			});
			blogs.sort((a, b) => a.title.localeCompare(b.title));
			return blogs.map(blog => `- [${blog.title}](${blog.url})`).join('\n');
		} catch (error) {
			console.error('Error parsing OPML file:', error);
			return "<!-- Error parsing blog subscriptions -->";
		}
	});
}


