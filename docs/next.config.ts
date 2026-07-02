import nextra from "nextra";

const withNextra = nextra({
  defaultShowCopyCode: true,
  search: { codeblocks: false },
  latex: true,
  readingTime: true,
});

export default withNextra({
  output: "export",
  typedRoutes: true,
  images: { unoptimized: true },
});
