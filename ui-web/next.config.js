/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  output: 'export',
  // Use `.next` during `next dev`; `out` only for production static export builds.
  distDir: process.env.NODE_ENV === 'production' ? 'out' : '.next',
  // Disable image optimization for static export
  images: {
    unoptimized: true,
  },
  // Ensure trailing slashes are consistent
  trailingSlash: true,
  webpack: (config) => {
    config.externals = [...(config.externals || []), { canvas: 'canvas' }];
    return config;
  },
};

module.exports = nextConfig;


