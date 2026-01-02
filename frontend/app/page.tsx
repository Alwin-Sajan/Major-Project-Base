// app/page.tsx
import Link from 'next/link';

export default function EntryPage() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: '20px' }}>
      <h1>Select Your Interface</h1>
      
      <div style={{ display: 'flex', gap: '20px' }}>

        {/* Button 1 */}
      <Link href="/simple-detection-ui">
        <button className="bg-green-600 hover:bg-green-700 text-white font-bold py-4 px-8 rounded-lg transition-colors duration-300">
          Simple Detection UI
        </button>
      </Link>

        {/* Button 2 */}
      <Link href="/ood-detection-ui">
        <button className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-8 rounded-lg transition-colors duration-300">
          OOD Detection UI
        </button>
      </Link>


      </div>
    </div>
  );
}