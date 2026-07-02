import "nextra-theme-docs/style.css";
import type { PropsWithChildren, ReactElement } from "react";
import { Footer, Layout, Navbar } from "nextra-theme-docs";
import { Head } from "nextra/components";
import { getPageMap } from "nextra/page-map";

// Polyfill Element for SSG workers (Nextra theme references it during server rendering)
if (typeof Element === "undefined") {
  (globalThis as Record<string, unknown>).Element = class Element {} as unknown;
}

export const metadata = {
  metadataBase: new URL("https://morphoclip.suxrobgm.net"),
  title: {
    default: "MorphoCLIP",
    template: "%s — MorphoCLIP",
  },
  description:
    "AI-powered matching of cell microscopy images with text descriptions of biological treatments",
  icons: {
    icon: [
      { url: "/favicon.svg", type: "image/svg+xml" },
      { url: "/favicon.png", type: "image/png", sizes: "32x32" },
    ],
  },
  openGraph: {
    title: "MorphoCLIP",
    description:
      "AI-powered matching of cell microscopy images with text descriptions of biological treatments",
    url: "https://morphoclip.suxrobgm.net",
    siteName: "MorphoCLIP",
    images: [{ url: "/og-image.png", width: 1200, height: 630, alt: "MorphoCLIP" }],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "MorphoCLIP",
    description:
      "AI-powered matching of cell microscopy images with text descriptions of biological treatments",
    images: ["/og-image.png"],
  },
};

const navbar = (
  <Navbar
    logo={
      <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
        <img src="/favicon.svg" alt="" width={24} height={24} />
        <span style={{ fontWeight: 700, fontSize: "1.1rem" }}>MorphoCLIP</span>
      </span>
    }
  />
);

const footer = <Footer>MIT {new Date().getFullYear()} © MorphoCLIP — CS7980 Spring 2026</Footer>;

export default async function RootLayout(props: PropsWithChildren): Promise<ReactElement> {
  const { children } = props;
  return (
    <html lang="en" dir="ltr" suppressHydrationWarning>
      <Head />
      <body>
        <Layout
          navbar={navbar}
          pageMap={await getPageMap()}
          footer={footer}
          editLink={null}
          sidebar={{ defaultMenuCollapseLevel: 2 }}
        >
          {children}
        </Layout>
      </body>
    </html>
  );
}
